"""Hype Vision - Okul Giris (yuz tanima) sunucusu.
Calistir:  venv\\Scripts\\python app.py   ->  http://localhost:8000
Modeller:  models/yunet.onnx, models/auraface/glintr100.onnx
Veri:      data/  (people/, gallery.npz, okul.db, settings.json, pending/, unknown/)
"""
import os, time, threading, json, re, shutil, sqlite3, hashlib, secrets, io, zipfile, base64, csv
import urllib.request, urllib.parse, queue
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response, JSONResponse, PlainTextResponse
import uvicorn

ROOT = Path(__file__).parent
MODELS, DATA = ROOT / "models", ROOT / "data"
PEOPLE, PENDING, UNKNOWN, SNAPS = DATA / "people", DATA / "pending", DATA / "unknown", DATA / "snaps"
GALLERY, SETTINGS, DB_PATH = DATA / "gallery.npz", DATA / "settings.json", DATA / "okul.db"
for d in (PEOPLE, PENDING, UNKNOWN, SNAPS): d.mkdir(parents=True, exist_ok=True)

# ================= yardimci =================
TR = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
def ascii_name(s): return (s or "").translate(TR)
def now_s(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_name(s):
    s = re.sub(r"[\\/:*?\"<>|]", "", s or "").strip()
    if not s or s in (".", ".."): raise HTTPException(400, "Isim bos")
    return s

def imread_u(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def imwrite_u(p, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92]); buf.tofile(str(p))

DEFAULTS = {
    "threshold": 0.40,      # bunun ustu: tanindi
    "low_threshold": 0.28,  # bununla threshold arasi: onay bekliyor; alti: bilinmeyen
    "vote_frames": 5,       # karar icin gereken kare
    "cooldown": 300,        # ayni kisi tekrar kayit (sn)
    "det_score": 0.8, "min_face": 60,
    "det_width": 960,      # tespit bu genislikte yapilir (yuz kesimi tam cozunurlukten)
    "queue_max": 75,
    "max_fps": 12,         # kamera basina saniyede en fazla bu kadar kare islenir (fazlasi atlanir, kuyruk birikmez)       # kuyruk: ~3 sn (25 fps)
    "keep_days": 90,        # gecis/mesaj kayitlari saklama
    "keep_unknown_days": 1, # bilinmeyen/onay fotolari saklama
    "wa_enabled": False, "wa_token": "", "wa_phone_id": "", "wa_verify": "hypevision",
    "wa_mode": "text",      # text | template
    "wa_template": "Sayın {veli}, öğrenciniz {ogrenci} okula giriş yaptı. Saat: {saat}",
    "wa_tpl_name": "", "wa_tpl_lang": "tr",
    "alarm_tel": "",
    "wa_provider": "whapi",  # whapi | meta
    "whapi_token": "",
    "wa_template_gec": "Sayın {veli}, öğrenciniz {ogrenci} okula GEÇ giriş yaptı. Saat: {saat}",
    "wa_template_cikis": "Sayın {veli}, öğrenciniz {ogrenci} okuldan ayrıldı. Saat: {saat}",
    "ders_baslangic": "08:30", "gelmedi_saat": "08:45", "tatiller": "", "hafta_sonu_kapali": True,
    "cam_down_sec": 60, "bot_enabled": False, "bot_keyword": "durum,nerede",
    "cameras": [],
    "cluster_threshold": 0.45,
    "keep_snap_days": 7,
    "okul_adi": "", "rapor_aktif": False, "rapor_gun": 0, "rapor_saat": "08:00", "rapor_tel": "", "rapor_email": "",
    "smtp_host": "", "smtp_port": 587, "smtp_user": "", "smtp_pass": "", "smtp_from": "",
    "central_token": "", "okullar": [],  # gecis anindaki CCTV kareleri (operator kontrolu icin)
    "margin": 0.08,       # en iyi aday ile ikinci aday arasinda en az bu fark yoksa "emin degil"
    "min_quality": 0.35,  # bunun altindaki kareler (bulanik / yandan / kucuk) oylamaya girmez  # bilinmeyen yuzleri ayni kisi sayma benzerligi
}
def load_settings():
    d = dict(DEFAULTS)
    if SETTINGS.exists(): d.update(json.loads(SETTINGS.read_text("utf-8")))
    for k in ("panel_user", "panel_pass"): d.pop(k, None)
    return d
S = load_settings()
def save_settings(): SETTINGS.write_text(json.dumps(S, ensure_ascii=False), "utf-8")

# ================= veritabani =================
DBL = threading.Lock()
def db():
    c = sqlite3.connect(DB_PATH, check_same_thread=False); c.row_factory = sqlite3.Row; return c
CONN = db()
CONN.executescript("""
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, time TEXT, name TEXT, score REAL, cam TEXT, sinif TEXT, liste TEXT, wa TEXT, kaynak TEXT);
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, time TEXT, tel TEXT, name TEXT, kind TEXT, text TEXT, status TEXT, wamid TEXT, err TEXT);
CREATE TABLE IF NOT EXISTS pending(id INTEGER PRIMARY KEY, time TEXT, kind TEXT, cand TEXT, score REAL, img TEXT, cam TEXT, status TEXT DEFAULT 'acik');
CREATE TABLE IF NOT EXISTS users(user TEXT PRIMARY KEY, pw TEXT, rol TEXT, sinif TEXT, ad TEXT);
CREATE TABLE IF NOT EXISTS tokens(tok TEXT PRIMARY KEY, user TEXT, exp REAL);
CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, time TEXT, user TEXT, action TEXT, target TEXT, detail TEXT);
""")
for _t, _c in (("events", "yon TEXT DEFAULT 'giris'"), ("events", "gec INTEGER DEFAULT 0"), ("pending", "emb BLOB"), ("pending", "grp TEXT"), ("events", "snap TEXT"), ("events", "dogrulama TEXT")):
    try: CONN.execute(f"ALTER TABLE {_t} ADD COLUMN " + _c); CONN.commit()
    except sqlite3.OperationalError: pass
def q(sql, args=(), one=False):
    with DBL:
        cur = CONN.execute(sql, args); CONN.commit()
        r = cur.fetchall()
    r = [dict(x) for x in r]
    return (r[0] if r else None) if one else r
def ins(sql, args):
    with DBL:
        cur = CONN.execute(sql, args); CONN.commit(); return cur.lastrowid

def hpw(pw, salt=None):
    salt = salt or secrets.token_hex(8)
    return salt + "$" + hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 100_000).hex()
def check_pw(pw, h): return hpw(pw, h.split("$")[0]) == h
if not q("SELECT 1 FROM users LIMIT 1"):
    ins("INSERT INTO users VALUES(?,?,?,?,?)", ("hype", hpw("hype"), "yonetici", "", "Yönetici"))

# eski csv'yi tasi
_old = DATA / "events.csv"
if _old.exists():
    with open(_old, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try: float(r["score"])
            except Exception: continue
            ins("INSERT INTO events(time,name,score,cam,sinif,liste,wa,kaynak) VALUES(?,?,?,?,?,?,?,?)",
                (r["time"], r["name"], float(r["score"]), r.get("cam", ""), r.get("sinif", ""), r.get("liste", "ogrenci"), r.get("wa", ""), "kamera"))
    _old.rename(DATA / "events_eski.csv")

# ================= kisiler =================
INFO_CACHE = {}
def read_info(name):
    if name not in INFO_CACHE:
        p = PEOPLE / name / "info.json"
        INFO_CACHE[name] = json.loads(p.read_text("utf-8")) if p.exists() else {}
    return INFO_CACHE[name]
def liste_of(name): return read_info(name).get("liste") or "ogrenci"

# ================= modeller =================
ARC_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
def norm_crop(img, kps):
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARC_DST, method=cv2.LMEDS)
    return cv2.warpAffine(img, M, (112, 112), borderValue=0)

def face_quality(crop, fc):
    """0..1 kalite: keskinlik, onden bakma, yuz boyu, tespit guveni."""
    kps = fc[4:14].reshape(5, 2); le, re_, nose = kps[0], kps[1], kps[2]
    ed = np.linalg.norm(re_ - le) + 1e-6
    front = max(0.0, 1 - abs(nose[0] - (le[0] + re_[0]) / 2) / ed * 2)        # yatay donus
    tilt = max(0.0, 1 - abs(np.degrees(np.arctan2(re_[1] - le[1], re_[0] - le[0]))) / 35)  # bas egikligi
    sharp = min(1.0, cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() / 120)
    size = min(1.0, float(fc[2]) / 110)
    return float(round((0.35 * front + 0.15 * tilt + 0.3 * sharp + 0.2 * size) * min(1.0, float(fc[14]) / 0.85), 3))

def face_thumb(img, fc, size=160):
    """Insanin gorebilecegi, biraz genis kesilmis yuz."""
    x, y, w, h = fc[:4]; cx, cy = x + w / 2, y + h / 2; s = max(w, h) * 1.6
    x0, y0 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
    x1, y1 = int(min(cx + s / 2, img.shape[1])), int(min(cy + s / 2, img.shape[0]))
    c = img[y0:y1, x0:x1]
    return cv2.resize(c, (size, size)) if c.size else None

class Engine:
    def __init__(self):
        self.det = cv2.FaceDetectorYN.create(str(MODELS / "yunet.onnx"), "", (320, 320), S["det_score"], 0.3, 5000)
        try: ort.preload_dlls()
        except Exception: pass
        prov = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()]
        self.rec = ort.InferenceSession(str(MODELS / "auraface" / "glintr100.onnx"), providers=prov)
        self.provider = self.rec.get_providers()[0]
        self.inp = self.rec.get_inputs()[0].name
        self.lock = threading.Lock()
        print("AuraFace provider:", self.provider)

    def detect(self, img):
        h, w = img.shape[:2]
        with self.lock:
            self.det.setInputSize((w, h)); self.det.setScoreThreshold(S["det_score"])
            _, faces = self.det.detect(img)
        return [] if faces is None else faces

    def embed(self, crops):
        if not crops: return np.zeros((0, 512), np.float32)
        blob = cv2.dnn.blobFromImages(crops, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        with self.lock: e = self.rec.run(None, {self.inp: blob})[0]
        return e / np.linalg.norm(e, axis=1, keepdims=True)
E = Engine()

class Gallery:
    """Her ogrencinin HER fotografi ayri saklanir. Canli yuz her fotografla karsilastirilir;
    kisi skoru = en iyi fotograf (+ 2. en iyi ile tutarlilik). Karar icin en iyi ve ikinci aday farki da bakilir."""
    def __init__(self):
        self.names, self.embs = [], np.zeros((0, 512), np.float32)          # kisi merkezleri (yardimci)
        self.P, self.owner, self.files = np.zeros((0, 512), np.float32), np.zeros(0, np.int32), []
        self.starts, self.counts = np.zeros(0, np.int64), np.zeros(0, np.int64)
        if GALLERY.exists():
            z = np.load(GALLERY, allow_pickle=False); self.names, self.embs = list(z["names"]), z["embs"]
            if "P" in z.files:
                self.P, self.owner = z["P"], z["owner"]; self.files = list(z["files"]) if "files" in z.files else []
                self._index()
        self.lock = threading.Lock()

    def _index(self):
        if len(self.owner):
            self.starts = np.r_[0, np.flatnonzero(np.diff(self.owner)) + 1].astype(np.int64)
            self.counts = np.diff(np.r_[self.starts, len(self.owner)]).astype(np.int64)

    def train(self):
        with self.lock:
            names, embs, P, owner, files, report = [], [], [], [], [], []
            for d in sorted(PEOPLE.iterdir()):
                if not d.is_dir(): continue
                ps = sorted(d.glob("*.jpg")); crops, fs = [], []
                for f in ps:
                    c = imread_u(f)
                    if c is not None: crops.append(c); fs.append(f.name)
                if not crops: continue
                e = E.embed(crops); m = e.mean(0); m /= np.linalg.norm(m)
                k = len(names); names.append(d.name); embs.append(m)
                P.append(e); owner += [k] * len(e); files += fs
                report.append({"name": d.name, "photos": len(crops), "min_sim": round(float((e @ m).min()), 3)})
            self.names = names
            self.embs = np.array(embs, np.float32) if embs else np.zeros((0, 512), np.float32)
            self.P = np.concatenate(P).astype(np.float32) if P else np.zeros((0, 512), np.float32)
            self.owner = np.array(owner, np.int32); self.files = files; self._index()
            np.savez(GALLERY, names=np.array(names), embs=self.embs, P=self.P, owner=self.owner, files=np.array(files))
            # --- galeri saglik kontrolu
            risk, pairs, suspicious = None, [], []
            if len(names) > 1:
                sim = self.embs @ self.embs.T; np.fill_diagonal(sim, -1)
                i, j = np.unravel_index(sim.argmax(), sim.shape)
                risk = {"a": names[i], "b": names[j], "sim": round(float(sim[i, j]), 3)}
                iu = np.triu_indices(len(names), 1); v = sim[iu]
                for t in np.argsort(-v)[:15]:
                    if v[t] < S["threshold"] - 0.1: break
                    pairs.append({"a": names[iu[0][t]], "b": names[iu[1][t]], "sim": round(float(v[t]), 3)})
                # yanlis etiket supheli fotolar: kendi kisisinden cok baskasina benzeyen
                full = self.P @ self.P.T
                for k in range(len(self.P)):
                    own = self.owner[k]; row = full[k].copy(); row[k] = -1
                    same = row[self.owner == own]; other = row[self.owner != own]
                    if not len(other): continue
                    o = int(np.argmax(np.where(self.owner != own, row, -2)))
                    own_best = float(same.max()) if len(same) else None
                    if (own_best is None and other.max() >= S["threshold"]) or (own_best is not None and other.max() > own_best):
                        suspicious.append({"name": names[own], "file": files[k], "own": None if own_best is None else round(own_best, 3),
                                           "other": names[self.owner[o]], "other_sim": round(float(other.max()), 3)})
            return {"people": report, "closest_pair": risk, "risky_pairs": pairs, "suspicious": suspicious[:50]}

    def match_full(self, embs):
        """Her yuz icin: en iyi kisi, skoru, ikinci kisi, skoru."""
        if len(self.names) == 0 or len(embs) == 0 or len(self.P) == 0:
            return [{"name": None, "score": 0.0, "name2": None, "score2": 0.0} for _ in range(len(embs))]
        sim = embs @ self.P.T                                            # (M, foto)
        t1 = np.maximum.reduceat(sim, self.starts, axis=1)               # kisi basi en iyi foto
        mask = sim >= t1[:, self.owner] - 1e-6
        sim2 = np.where(mask, -1.0, sim)
        t2 = np.maximum.reduceat(sim2, self.starts, axis=1)              # kisi basi 2. en iyi foto
        multi = self.counts[None, :] > 1
        score = np.where(multi & (t2 > -1), 0.8 * t1 + 0.2 * np.maximum(t2, t1 - 0.15), t1)
        out = []
        for r in score:
            o = np.argsort(-r)[:2]
            out.append({"name": self.names[o[0]], "score": float(r[o[0]]),
                        "name2": self.names[o[1]] if len(o) > 1 else None, "score2": float(r[o[1]]) if len(o) > 1 else 0.0})
        return out

    def match(self, embs):
        return [(m["name"], m["score"]) for m in self.match_full(embs)]

def decide(m, q=1.0):
    """Tek karede kabul: esik + ikinci adayla fark + kalite. Donus: (isim|None, sebep)."""
    if m["name"] is None: return None, "galeri bos"
    if q < S.get("min_quality", 0.35): return None, "kalite"
    if m["score"] < S["threshold"]: return None, "esik"
    if m["score"] - m["score2"] < S.get("margin", 0.08): return None, "fark"
    return m["name"], "ok"

G = Gallery()
def train_bg(): threading.Thread(target=G.train, daemon=True).start()

# ================= WhatsApp =================
def wa_phone(tel):
    d = re.sub(r"\D", "", tel or "")
    if d.startswith("0"): d = "9" + d
    elif len(d) == 10: d = "90" + d
    return d

def wa_vars(info, name, t): return {"veli": info.get("veli_ad") or "Veli", "ogrenci": name, "saat": t[11:16]}

def wa_ready():
    return bool(S["whapi_token"]) if S["wa_provider"] == "whapi" else bool(S["wa_token"] and S["wa_phone_id"])

def wa_send(tel_raw, name, kind, vars_or_text, ev_id=None, force_text=False):
    """Mesaji kaydeder, gonderir, messages tablosunu gunceller. Donus: durum metni."""
    tel = wa_phone(tel_raw)
    text = vars_or_text if isinstance(vars_or_text, str) else S["wa_template"].format(**vars_or_text)
    if not tel: return "numara yok"
    if not (S["wa_enabled"] or force_text) or not wa_ready(): return "kapali"
    mid = ins("INSERT INTO messages(time,tel,name,kind,text,status) VALUES(?,?,?,?,?,?)", (now_s(), tel, name, kind, text, "gonderiliyor"))
    def send():
        if S["wa_provider"] == "whapi":
            url, tok = "https://gate.whapi.cloud/messages/text", S["whapi_token"]
            body = {"to": tel, "body": text}
        elif S["wa_mode"] == "template" and S["wa_tpl_name"] and not isinstance(vars_or_text, str):
            v = vars_or_text
            body = {"messaging_product": "whatsapp", "to": tel, "type": "template",
                    "template": {"name": S["wa_tpl_name"], "language": {"code": S["wa_tpl_lang"] or "tr"},
                                 "components": [{"type": "body", "parameters": [
                                     {"type": "text", "text": v["veli"]}, {"type": "text", "text": v["ogrenci"]},
                                     {"type": "text", "text": v["saat"]}]}]}}
        else:
            body = {"messaging_product": "whatsapp", "to": tel, "type": "text", "text": {"body": text}}
        if S["wa_provider"] != "whapi":
            url, tok = f"https://graph.facebook.com/v21.0/{S['wa_phone_id']}/messages", S["wa_token"]
        req = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Authorization": "Bearer " + tok, "Content-Type": "application/json", "Accept": "application/json",
                                      "User-Agent": "HypeVision/1.0 (+okul-giris)"})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=15).read())
            if S["wa_provider"] == "whapi":
                if r.get("sent") is False: raise Exception(json.dumps(r)[:300])
                wamid = (r.get("message") or {}).get("id", "") or r.get("id", "")
            else:
                wamid = (r.get("messages") or [{}])[0].get("id", "")
            q("UPDATE messages SET status='gonderildi', wamid=? WHERE id=?", (wamid, mid)); st = "gonderildi"
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", "ignore")[:300]
            q("UPDATE messages SET status='hata', err=? WHERE id=?", (err, mid)); st = "hata"
        except Exception as e:
            q("UPDATE messages SET status='hata', err=? WHERE id=?", (str(e)[:300], mid)); st = "hata"
        if ev_id: q("UPDATE events SET wa=? WHERE id=?", (st, ev_id))
    threading.Thread(target=send, daemon=True).start()
    return "gonderiliyor"

# ================= kurallar (saat / tatil) =================
def today_s(): return datetime.now().strftime("%Y-%m-%d")
def tatil_set(): return {x.strip() for x in re.split(r"[,\s;]+", S.get("tatiller", "")) if x.strip()}
def is_off_day(day=None):
    d = datetime.strptime(day or today_s(), "%Y-%m-%d")
    return (S.get("hafta_sonu_kapali", True) and d.weekday() >= 5) or d.strftime("%Y-%m-%d") in tatil_set()
def is_late(t):  # t: "YYYY-mm-dd HH:MM:SS"
    return bool(S.get("ders_baslangic")) and t[11:16] > S["ders_baslangic"]

# ================= olay kaydi =================
LAST_SEEN = {}
ALARMS = []      # kara liste alarm akisi
CAM_ALERTS = []  # kamera koptu uyarilari

def save_snap(eid, frame, fc, face=None):
    """Gecis anindaki CCTV karesini (kutu cizili) ve yuz kesitini sakla."""
    f = frame.copy(); x, y, w, h = map(int, fc[:4])
    cv2.rectangle(f, (x, y), (x + w, y + h), (0, 220, 0), max(2, f.shape[1] // 400))
    if f.shape[1] > 1280: f = cv2.resize(f, None, fx=1280 / f.shape[1], fy=1280 / f.shape[1], interpolation=cv2.INTER_AREA)
    imwrite_u(SNAPS / f"{eid}.jpg", f)
    if face is None: face = face_thumb(frame, fc, 220)
    if face is not None: imwrite_u(SNAPS / f"{eid}_f.jpg", face)
    q("UPDATE events SET snap=? WHERE id=?", (f"{eid}", eid))

def log_event(name, score, cam_name, yon="giris", kaynak="kamera", force=False, snap=None):
    now = time.time(); key = (name, yon)
    if not force and now - LAST_SEEN.get(key, 0) < S["cooldown"]: return None
    LAST_SEEN[key] = now
    info, t = read_info(name), now_s()
    lst = liste_of(name)
    gec = 1 if (yon == "giris" and lst == "ogrenci" and is_late(t)) else 0
    eid = ins("INSERT INTO events(time,name,score,cam,sinif,liste,wa,kaynak,yon,gec) VALUES(?,?,?,?,?,?,?,?,?,?)",
              (t, name, round(score, 3), cam_name, info.get("sinif", ""), lst, "", kaynak, yon, gec))
    if snap is not None:
        try: save_snap(eid, *snap)
        except Exception as e: print("snap hatasi:", e)
    if lst == "kara":
        ALARMS.insert(0, {"id": eid, "time": t, "name": name, "not": info.get("not", ""), "cam": cam_name}); del ALARMS[50:]
        wa = wa_send(S["alarm_tel"], name, "alarm", f"⚠ KARA LİSTE: {name} {cam_name} kamerasında görüldü. Saat {t[11:16]}. {info.get('not', '')}", eid, force_text=True)
    elif lst == "beyaz": wa = "-"
    elif lst == "ziyaretci":
        wa = "-"
        if yon == "cikis" and info.get("durum") != "cikti":
            info["durum"] = "cikti"; info["cikis"] = t[:16]
            (PEOPLE / name / "info.json").write_text(json.dumps(info, ensure_ascii=False), "utf-8"); INFO_CACHE.pop(name, None)
    elif is_off_day(): wa = "tatil"
    else:
        v = wa_vars(info, name, t)
        if yon == "cikis": wa = wa_send(info.get("veli_tel"), name, "cikis", S["wa_template_cikis"].format(**v), eid)
        elif gec: wa = wa_send(info.get("veli_tel"), name, "gec", S["wa_template_gec"].format(**v), eid)
        else: wa = wa_send(info.get("veli_tel"), name, "geldi", v, eid)
    q("UPDATE events SET wa=? WHERE id=? AND wa=''", (wa, eid))
    return eid

# ---- bilinmeyen yuz gruplama (ayni kisi -> ayni rastgele id)
CLU_LOCK = threading.Lock()
def new_gid(): return "K-" + secrets.token_hex(2).upper()

def unk_centroids():
    rows = q("SELECT grp, emb FROM pending WHERE kind='bilinmeyen' AND emb IS NOT NULL AND grp IS NOT NULL")
    acc = {}
    for r in rows:
        e = np.frombuffer(r["emb"], np.float32)
        a = acc.setdefault(r["grp"], [np.zeros(512, np.float32), 0]); a[0] = a[0] + e; a[1] += 1
    gids = list(acc)
    if not gids: return [], np.zeros((0, 512), np.float32)
    C = np.stack([acc[g][0] / acc[g][1] for g in gids]); C /= np.linalg.norm(C, axis=1, keepdims=True)
    return gids, C

def assign_group(emb):
    gids, C = unk_centroids()
    if len(gids):
        sim = C @ emb; k = int(sim.argmax())
        if sim[k] >= S.get("cluster_threshold", 0.45): return gids[k]
    return new_gid()

def add_pending(kind, cand, score, thumb, crop, cam):
    emb = E.embed([crop])[0].astype(np.float32)
    with CLU_LOCK:
        grp = assign_group(emb) if kind == "bilinmeyen" else None
        pid = ins("INSERT INTO pending(time,kind,cand,score,img,cam,emb,grp) VALUES(?,?,?,?,?,?,?,?)",
                  (now_s(), kind, cand or "", round(score, 3), "", cam, emb.tobytes(), grp))
    folder = PENDING if kind == "onay" else UNKNOWN
    imwrite_u(folder / f"{pid}.jpg", thumb); imwrite_u(folder / f"{pid}_crop.jpg", crop)
    q("UPDATE pending SET img=? WHERE id=?", (f"{pid}.jpg", pid))

def recluster():
    """Tum bilinmeyenleri bastan grupla (eksik embedding'leri de hesaplar)."""
    with CLU_LOCK:
        rows = q("SELECT id, emb FROM pending WHERE kind='bilinmeyen' ORDER BY id")
        ids, embs = [], []
        for r in rows:
            e = r["emb"]
            if e is None:
                c = imread_u(UNKNOWN / f"{r['id']}_crop.jpg")
                if c is None: continue
                e = E.embed([c])[0].astype(np.float32).tobytes(); q("UPDATE pending SET emb=? WHERE id=?", (e, r["id"]))
            ids.append(r["id"]); embs.append(np.frombuffer(e, np.float32))
        if not ids: return 0
        X = np.stack(embs); th = S.get("cluster_threshold", 0.45)
        cents, members = [], []
        for i, e in enumerate(X):  # sirali merkez-tabanli gruplama
            if cents:
                C = np.stack(cents); sim = C @ e; k = int(sim.argmax())
                if sim[k] >= th:
                    members[k].append(i); m = X[members[k]].mean(0); cents[k] = m / np.linalg.norm(m); continue
            cents.append(e / np.linalg.norm(e)); members.append([i])
        # birbirine cok yakin merkezleri birlestir
        merged = True
        while merged and len(cents) > 1:
            merged = False; C = np.stack(cents); sim = C @ C.T; np.fill_diagonal(sim, -1)
            a, b = np.unravel_index(sim.argmax(), sim.shape)
            if sim[a, b] >= th:
                members[a] += members[b]; m = X[members[a]].mean(0); cents[a] = m / np.linalg.norm(m)
                del cents[b], members[b]; merged = True
        for mem in members:
            g = new_gid()
            for i in mem: q("UPDATE pending SET grp=? WHERE id=?", (g, ids[i]))
        return len(members)

# ================= kamera + takip/oylama =================
def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    i = ix * iy; return i / (aw * ah + bw * bh - i + 1e-6)

def vote_result(tr):
    """Oylama: son N gecerli karenin en az %70'i ayni kisi olmali ve baska bir kisiye HIC oy cikmamis olmali."""
    n = int(S["vote_frames"])
    if tr.done or len(tr.votes) < n: return None
    c = Counter(v[0] for v in tr.votes)
    top, cnt = c.most_common(1)[0]
    if not top or cnt < 0.7 * len(tr.votes): return None
    others = [k for k in c if k and k != top]
    if others: return None  # ayni izde iki farkli isim -> asla karar verme, onaya dus
    return top

class Track:
    _n = 0
    def __init__(self, box):
        Track._n += 1; self.id = Track._n; self.box = box; self.votes = []; self.last = time.time(); self.sc = 0.0
        self.done = False; self.decided = None; self.best = (0.0, None, None, None)  # score, name, thumb, crop
    def label(self):
        if self.decided: return self.decided
        if not self.votes: return None
        return Counter(v[0] for v in self.votes).most_common(1)[0][0]

# Tum kameralar tek GPU kuyruguna yazar; tek isleyici sirayla isler.
GQ = queue.Queue(maxsize=int(S["queue_max"]) * 4)
PERF = {"tespit": 0.0, "tanima": 0.0, "toplam": 0.0, "fps": 0.0, "dropped": 0, "lag": 0.0}

def open_webcam(i):
    """USB / dahili webcam: Windows'ta DirectShow (hizli acilir), Linux'ta V4L2; 1280x720 MJPG ister."""
    import platform
    be = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(i, be)
    if not cap.isOpened(): cap = cv2.VideoCapture(i)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720); cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

class Camera:
    """Tek bir RTSP kaynagi: okur, ortak kuyruga atar; kendi izlerini ve son goruntusunu tutar."""
    def __init__(self, cfg):
        self.cfg = cfg; self.frame = None; self.annot = None; self.snap = None
        self.running = False; self.in_fps = 0.0; self.fps = 0.0; self.status = "kapali"; self.tracks = []
        self.lock = threading.Lock(); self.gen = 0; self.last_frame = 0.0; self.started = 0.0
        self.down = False; self.alerted = False; self._n = 0; self._t0 = time.time(); self._last_q = 0.0

    @property
    def id(self): return self.cfg["id"]
    @property
    def name(self): return self.cfg.get("ad") or self.cfg["id"]
    @property
    def yon(self): return self.cfg.get("yon") or "giris"
    def url_short(self): return re.sub(r"//[^@/]*@", "//", self.cfg.get("url") or "")

    def start(self):
        self.stop(); self.running = True; self.status = "baglaniyor"; self.gen += 1; self.started = time.time()
        threading.Thread(target=self._reader, args=(self.gen,), daemon=True).start()

    def stop(self):
        self.running = False; self.gen += 1
        self.frame = None; self.annot = None; self.status = "kapali"; self.tracks = []

    def _reader(self, g):
        url = self.cfg["url"]; src = int(url) if url.isdigit() else url
        t0, n = time.time(), 0
        while self.running and g == self.gen:
            cap = open_webcam(src) if isinstance(src, int) else cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if not cap.isOpened(): self.status = "baglanamadi"; time.sleep(3); continue
            self.status = "canli"
            while self.running and g == self.gen:
                ok, f = cap.read()
                if not ok: self.status = "akis koptu"; break
                ts = time.time(); self.last_frame = ts
                with self.lock: self.frame = f
                n += 1
                if ts - t0 > 1: self.in_fps = n / (ts - t0); t0, n = ts, 0
                mf = float(S.get("max_fps", 12) or 0)
                if mf and ts - self._last_q < 1.0 / mf: continue  # bu kareyi isleme (canli gorunum son kareyi gosterir)
                self._last_q = ts
                try: GQ.put_nowait((self, g, ts, f))
                except queue.Full:
                    try: GQ.get_nowait(); PERF["dropped"] += 1
                    except queue.Empty: pass
                    try: GQ.put_nowait((self, g, ts, f))
                    except queue.Full: pass
            cap.release(); time.sleep(1)

    def finish(self, tr):
        if tr.done or len(tr.votes) < 2: return
        sc, name, thumb, crop = tr.best
        if thumb is None: return
        add_pending("onay" if sc >= S["low_threshold"] else "bilinmeyen", name, sc, thumb, crop, self.name)

    def process(self, ts, f):
        dw = int(S["det_width"]); scale = 1.0; img = f
        if f.shape[1] > dw: scale = dw / f.shape[1]; img = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ta = time.perf_counter(); faces = E.detect(img); tb = time.perf_counter()
        used, items = set(), []
        for fc in faces:
            fc = fc.copy(); fc[:14] /= scale; box = tuple(fc[:4])
            if min(fc[2], fc[3]) < S["min_face"]: items.append((fc, None)); continue
            tr = max((t for t in self.tracks if id(t) not in used), key=lambda t: iou(t.box, box), default=None)
            if tr is None or iou(tr.box, box) < 0.2: tr = Track(box); self.tracks.append(tr)
            used.add(id(tr)); tr.box = box; tr.last = ts; items.append((fc, tr))
        need = [(fc, tr) for fc, tr in items if tr is not None and not tr.done]
        crops = [norm_crop(f, fc[4:14].reshape(5, 2)) for fc, _ in need]
        res = dict(zip((id(tr) for _, tr in need), zip(crops, G.match_full(E.embed(crops)), (face_quality(c, fc) for c, (fc, _) in zip(crops, need)))))
        tc = time.perf_counter()
        vis = f.copy()
        for fc, tr in items:
            box = tuple(fc[:4])
            if tr is None:
                x, y, w, h = map(int, box); cv2.rectangle(vis, (x, y), (x + w, y + h), (150, 150, 150), 1); continue
            if id(tr) in res:
                crop, m, qv = res[id(tr)]; name, sc = m["name"], m["score"]
                acc, why = decide(m, qv); tr.sc = sc; tr.why = why
                if why != "kalite":  # kalitesiz kare oy kullanmaz
                    tr.votes.append((acc, sc)); tr.votes = tr.votes[-15:]
                    if sc > tr.best[0]: tr.best = (sc, name, face_thumb(f, fc), crop)
                top = vote_result(tr)
                if top:
                    tr.done = True; tr.decided = top
                    log_event(top, float(np.mean([v[1] for v in tr.votes if v[0] == top])), self.name, self.yon, snap=(f, fc))
            self._draw(vis, box, tr)
        for tr in [t for t in self.tracks if ts - t.last > 1.5]:
            self.finish(tr); self.tracks.remove(tr)
        cv2.putText(vis, f"{ascii_name(self.name)} ({'GIRIS' if self.yon == 'giris' else 'CIKIS'})", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        self.annot = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 78])[1].tobytes()
        return tb - ta, tc - tb

    def _draw(self, vis, box, tr):
        x, y, w, h = map(int, box); lab = tr.label(); sc = tr.sc
        if lab:
            lst = liste_of(lab)
            col = (0, 0, 230) if lst == "kara" else (230, 140, 0) if lst == "beyaz" else (200, 70, 160) if lst == "ziyaretci" else (0, 190, 0)
            pre = "KARA LISTE: " if lst == "kara" else "[B] " if lst == "beyaz" else "[Z] " if lst == "ziyaretci" else ""
            txt = f"{pre}{ascii_name(lab)} {sc:.2f}" + (" OK" if tr.done else f" {len(tr.votes)}/{int(S['vote_frames'])}")
        else:
            col = (0, 165, 255); why = getattr(tr, "why", "")
            txt = f"? {sc:.2f}" + {"kalite": " (kalite)", "fark": " (benzer var)"}.get(why, "")
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 4 if lab and liste_of(lab) == "kara" else 2)
        cv2.rectangle(vis, (x, y - 26), (x + max(w, 11 * len(txt)), y), col, -1)
        cv2.putText(vis, txt, (x + 4, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

CAMS = {}  # id -> Camera

def worker():
    t0, n = time.time(), 0; a = 0.9
    while True:
        try: cam, g, ts, f = GQ.get(timeout=0.5)
        except queue.Empty: continue
        if g != cam.gen or not cam.running: continue
        t = time.perf_counter()
        try: d, r = cam.process(ts, f)
        except Exception as e: print("isleme hatasi:", e); continue
        tot = time.perf_counter() - t
        for k, v in (("tespit", d), ("tanima", r), ("toplam", tot)): PERF[k] = a * PERF[k] + (1 - a) * v * 1000
        PERF["lag"] = time.time() - ts
        cam._n += 1; n += 1; now = time.time()
        if now - cam._t0 > 1: cam.fps = cam._n / (now - cam._t0); cam._t0, cam._n = now, 0
        if now - t0 > 1: PERF["fps"] = n / (now - t0); t0, n = now, 0
threading.Thread(target=worker, daemon=True).start()

def cams_cfg(): return S.setdefault("cameras", [])
def apply_cameras():
    ids = {c["id"] for c in cams_cfg()}
    for cid in list(CAMS):
        if cid not in ids: CAMS.pop(cid).stop()
    for c in cams_cfg():
        cam = CAMS.get(c["id"])
        if cam is None: cam = CAMS[c["id"]] = Camera(c)
        changed = cam.cfg.get("url") != c.get("url") or not cam.running
        cam.cfg = c
        if c.get("aktif", True):
            if changed: cam.start()
        elif cam.running: cam.stop()

def watchdog():
    """Kamera koptu uyarisi: aktif kamera cam_down_sec boyunca kare vermezse."""
    while True:
        time.sleep(10)
        lim = float(S.get("cam_down_sec", 60))
        for cam in list(CAMS.values()):
            if not cam.running: cam.down = False; cam.alerted = False; continue
            ref = cam.last_frame or cam.started
            if time.time() - ref > lim:
                cam.down = True
                if not cam.alerted:
                    cam.alerted = True; t = now_s()
                    CAM_ALERTS.insert(0, {"time": t, "cam": cam.name, "msg": "görüntü yok"}); del CAM_ALERTS[50:]
                    wa_send(S["alarm_tel"], cam.name, "kamera", f"⚠ Kamera uyarısı: {cam.name} {int(lim)} sn'dir görüntü vermiyor. ({t[11:16]})", force_text=True)
                    q("INSERT INTO audit(time,user,action,target,detail) VALUES(?,?,?,?,?)", (t, "sistem", "kamera koptu", cam.name, ""))
            elif cam.down:
                cam.down = False; cam.alerted = False; t = now_s()
                CAM_ALERTS.insert(0, {"time": t, "cam": cam.name, "msg": "düzeldi"})
                wa_send(S["alarm_tel"], cam.name, "kamera", f"✓ {cam.name} kamerası tekrar çalışıyor. ({t[11:16]})", force_text=True)
threading.Thread(target=watchdog, daemon=True).start()

# ================= temizlik (saklama suresi) =================
def cleanup():
    while True:
        try:
            cut = (datetime.now() - timedelta(days=S["keep_days"])).strftime("%Y-%m-%d")
            q("DELETE FROM events WHERE time < ?", (cut,)); q("DELETE FROM messages WHERE time < ?", (cut,))
            ucut = (datetime.now() - timedelta(days=S["keep_unknown_days"])).strftime("%Y-%m-%d %H:%M:%S")
            for r in q("SELECT id,kind FROM pending WHERE time < ?", (ucut,)): drop_pending(r["id"], r["kind"])
            q("DELETE FROM tokens WHERE exp < ?", (time.time(),))
            scut = (datetime.now() - timedelta(days=S.get("keep_snap_days", 7))).strftime("%Y-%m-%d")
            for r in q("SELECT id FROM events WHERE snap IS NOT NULL AND time < ?", (scut,)):
                for suf in ("", "_f"): (SNAPS / f"{r['id']}{suf}.jpg").unlink(missing_ok=True)
                q("UPDATE events SET snap=NULL WHERE id=?", (r["id"],))
        except Exception as e: print("temizlik hatasi:", e)
        time.sleep(3600)

def drop_pending(pid, kind=None):
    for folder in (PENDING, UNKNOWN):
        for suf in ("", "_crop"): (folder / f"{pid}{suf}.jpg").unlink(missing_ok=True)
    q("DELETE FROM pending WHERE id=?", (pid,))
threading.Thread(target=cleanup, daemon=True).start()

# ================= yetki =================
app = FastAPI()
ROLES = {"yonetici": 3, "mudur_yrd": 2, "ogretmen": 1}
OPEN = {"/api/login", "/webhook/whatsapp", "/api/central/summary"}

@app.middleware("http")
async def auth(request: Request, call_next):
    p = request.url.path
    if (p.startswith("/api/") or p.startswith("/media/")) and p not in OPEN:
        t = q("SELECT u.* FROM tokens t JOIN users u ON u.user=t.user WHERE t.tok=? AND t.exp>?",
              (request.cookies.get("tok", ""), time.time()), one=True)
        if not t: return JSONResponse({"detail": "giris gerekli"}, 401)
        request.state.u = t
    return await call_next(request)

def need(request, rol):
    if ROLES.get(request.state.u["rol"], 0) < ROLES[rol]: raise HTTPException(403, "Bu islem icin yetkin yok")
def my_class(request):
    u = request.state.u; return u["sinif"] if u["rol"] == "ogretmen" else None
def visible(request, name):
    c = my_class(request); return c is None or read_info(name).get("sinif") == c

@app.post("/api/login")
def login(d: dict):
    u = q("SELECT * FROM users WHERE user=?", (d.get("user", ""),), one=True)
    if not u or not check_pw(d.get("pass", ""), u["pw"]): raise HTTPException(401, "Kullanıcı adı veya şifre hatalı")
    t = secrets.token_hex(16); days = 30 if d.get("remember") else 1
    ins("INSERT INTO tokens VALUES(?,?,?)", (t, u["user"], time.time() + days * 86400))
    ins("INSERT INTO audit(time,user,action,target,detail) VALUES(?,?,?,?,?)", (now_s(), u["user"], "giriş yaptı", "", ""))
    r = JSONResponse({"ok": True})
    r.set_cookie("tok", t, httponly=True, samesite="lax", max_age=days * 86400 if d.get("remember") else None)
    return r

@app.post("/api/logout")
def logout(request: Request): q("DELETE FROM tokens WHERE tok=?", (request.cookies.get("tok", ""),)); return {"ok": True}

@app.get("/api/me")
def me(request: Request):
    u = request.state.u; return {"user": u["user"], "rol": u["rol"], "sinif": u["sinif"], "ad": u["ad"]}

# ---- kullanicilar
@app.get("/api/users")
def users(request: Request):
    need(request, "yonetici"); return q("SELECT user,rol,sinif,ad FROM users ORDER BY rol,user")

@app.post("/api/users")
def save_user(request: Request, d: dict):
    need(request, "yonetici")
    user = (d.get("user") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{2,32}", user): raise HTTPException(400, "Kullanıcı adı: harf/rakam, 2-32 karakter")
    rol = d.get("rol") if d.get("rol") in ROLES else "ogretmen"
    old = q("SELECT * FROM users WHERE user=?", (user,), one=True)
    if old and old["rol"] == "yonetici" and rol != "yonetici" and len(q("SELECT 1 FROM users WHERE rol='yonetici'")) == 1:
        raise HTTPException(400, "Son yöneticinin rolü değiştirilemez")
    pw = hpw(d["pass"]) if d.get("pass") else (old["pw"] if old else None)
    if not pw: raise HTTPException(400, "Yeni kullanıcı için şifre gerekli")
    q("INSERT OR REPLACE INTO users VALUES(?,?,?,?,?)", (user, pw, rol, (d.get("sinif") or "").strip(), (d.get("ad") or "").strip()))
    if d.get("pass") and old: q("DELETE FROM tokens WHERE user=? AND tok<>?", (user, request.cookies.get("tok", "")))
    audit(request, "kullanıcı kaydedildi", user, rol); return {"ok": True}

@app.delete("/api/users/{user}")
def del_user(request: Request, user: str):
    need(request, "yonetici")
    if user == request.state.u["user"]: raise HTTPException(400, "Kendini silemezsin")
    q("DELETE FROM users WHERE user=?", (user,)); q("DELETE FROM tokens WHERE user=?", (user,)); audit(request, "kullanıcı silindi", user); return {"ok": True}

@app.post("/api/password")
def change_pw(request: Request, d: dict):
    u = request.state.u
    if not check_pw(d.get("old", ""), u["pw"]): raise HTTPException(400, "Mevcut şifre yanlış")
    if len(d.get("new", "")) < 4: raise HTTPException(400, "Yeni şifre en az 4 karakter")
    q("UPDATE users SET pw=? WHERE user=?", (hpw(d["new"]), u["user"])); return {"ok": True}

# ================= sayfalar =================
@app.get("/", response_class=HTMLResponse)
def index(): return HTMLResponse((ROOT / "index.html").read_text("utf-8"), headers={"Cache-Control": "no-cache"})

@app.get("/favicon.ico")
def favicon(): return Response((ROOT / "logo" / "icon-64.png").read_bytes(), media_type="image/png")

@app.get("/manifest.json")
def manifest():
    return JSONResponse({"name": "Hype Vision · Okul Giriş", "short_name": "Hype Vision", "start_url": "/", "display": "standalone",
                         "background_color": "#ffffff", "theme_color": "#06b6d4",
                         "icons": [{"src": "/logo/icon-180.png", "sizes": "180x180", "type": "image/png"},
                                   {"src": "/logo/icon-512.png", "sizes": "512x512", "type": "image/png"}]})

@app.get("/logo/{f}")
def logo(f: str):
    p = ROOT / "logo" / safe_name(f)
    if not p.exists(): raise HTTPException(404)
    return Response(p.read_bytes(), media_type="image/png")

@app.get("/api/status")
def status(request: Request):
    pub = {k: v for k, v in S.items() if k not in ("wa_token", "whapi_token", "cameras", "smtp_pass", "central_token", "okullar")} | \
          {"wa_token_set": bool(S["wa_token"]), "whapi_token_set": bool(S["whapi_token"]), "smtp_pass_set": bool(S.get("smtp_pass")), "central_token_set": bool(S.get("central_token"))}
    cams = [{"id": c.id, "ad": c.name, "yon": c.yon, "status": c.status, "aktif": c.cfg.get("aktif", True),
             "in_fps": round(c.in_fps, 1), "fps": round(c.fps, 1), "down": c.down} for c in CAMS.values()]
    absent = None
    if S.get("gelmedi_saat") and datetime.now().strftime("%H:%M") >= S["gelmedi_saat"] and not is_off_day():
        seen = {r["name"] for r in q("SELECT DISTINCT name FROM events WHERE time LIKE ? AND yon='giris'", (today_s() + "%",))}
        n = sum(1 for d in PEOPLE.iterdir() if d.is_dir() and liste_of(d.name) == "ogrenci" and d.name not in seen and visible(request, d.name))
        if n: absent = {"count": n, "saat": S["gelmedi_saat"], "day": today_s()}
    return {"cams": cams, "provider": E.provider, "perf": {k: round(v, 1) for k, v in PERF.items()},
            "queue": GQ.qsize(), "queue_max": GQ.maxsize,
            "gallery": len(G.names), "settings": pub, "off_day": is_off_day(), "absent": absent,
            "pending": len([r for r in q("SELECT cand FROM pending WHERE kind='onay'") if visible(request, r["cand"])]),
            "unknown": q("SELECT COUNT(*) n FROM pending WHERE kind='bilinmeyen'", one=True)["n"],
            "alarms": ALARMS[:5], "cam_alerts": CAM_ALERTS[:5],
            "visit_alerts": [a for a in VISIT_ALERTS if read_info(a["name"]).get("durum") == "iceride"][:5],
            "visitors_in": sum(1 for d in PEOPLE.iterdir() if d.is_dir() and read_info(d.name).get("liste") == "ziyaretci" and read_info(d.name).get("durum") == "iceride")}

@app.post("/api/settings")
def settings(request: Request, d: dict):
    need(request, "yonetici")
    for k in ("threshold", "low_threshold", "cooldown", "det_score", "min_face", "vote_frames", "det_width", "queue_max",
              "keep_days", "keep_unknown_days", "keep_snap_days", "cam_down_sec", "max_fps", "margin", "min_quality", "cluster_threshold"):
        if d.get(k) not in (None, ""): S[k] = float(d[k])
    for k in ("wa_phone_id", "wa_template", "wa_template_gec", "wa_template_cikis", "wa_tpl_name", "wa_tpl_lang", "wa_mode",
              "alarm_tel", "wa_verify", "wa_provider", "ders_baslangic", "gelmedi_saat", "tatiller", "bot_keyword",
              "okul_adi", "rapor_saat", "rapor_tel", "rapor_email", "smtp_host", "smtp_user", "smtp_from"):
        if k in d: S[k] = str(d[k]).strip()
    if d.get("wa_token"): S["wa_token"] = str(d["wa_token"]).strip()
    if d.get("whapi_token"): S["whapi_token"] = str(d["whapi_token"]).strip()
    if d.get("smtp_pass"): S["smtp_pass"] = str(d["smtp_pass"])
    for k in ("rapor_gun", "smtp_port"):
        if d.get(k) not in (None, ""): S[k] = int(d[k])
    for k in ("wa_enabled", "hafta_sonu_kapali", "bot_enabled", "rapor_aktif"):
        if k in d: S[k] = bool(d[k])
    save_settings(); audit(request, "ayar değişti", "", ", ".join(k for k in d if "token" not in k)); return {"ok": True}

# ---- kameralar
@app.get("/api/cameras")
def cameras(request: Request):
    full = ROLES[request.state.u["rol"]] >= 2
    out = []
    for c in cams_cfg():
        cam = CAMS.get(c["id"])
        out.append({**c, "url": c["url"] if full else "", "url_short": cam.url_short() if cam else "",
                    "status": cam.status if cam else "kapali", "down": cam.down if cam else False,
                    "in_fps": round(cam.in_fps, 1) if cam else 0, "fps": round(cam.fps, 1) if cam else 0})
    return out

@app.post("/api/cameras")
def save_camera(request: Request, d: dict):
    need(request, "mudur_yrd")
    ad = (d.get("ad") or "").strip(); url = (d.get("url") or "").strip()
    if not ad or not url: raise HTTPException(400, "Kamera adı ve RTSP adresi gerekli")
    cid = d.get("id") or secrets.token_hex(4)
    c = {"id": cid, "ad": ad, "url": url, "yon": "cikis" if d.get("yon") == "cikis" else "giris", "aktif": bool(d.get("aktif", True))}
    lst = cams_cfg(); i = next((k for k, x in enumerate(lst) if x["id"] == cid), None)
    if i is None: lst.append(c)
    else: lst[i] = c
    save_settings(); apply_cameras(); audit(request, "kamera kaydedildi", ad, c["yon"]); return {"ok": True, "id": cid}

@app.delete("/api/cameras/{cid}")
def del_camera(request: Request, cid: str):
    need(request, "mudur_yrd")
    S["cameras"] = [c for c in cams_cfg() if c["id"] != cid]; save_settings(); apply_cameras()
    audit(request, "kamera silindi", cid); return {"ok": True}

@app.get("/api/stream/{cid}")
async def stream(request: Request, cid: str):
    """MJPEG akisi. Async: tarayici kapaninca / sayfa yenilenince hemen biter (is parcacigi tutmaz)."""
    import asyncio
    cam = CAMS.get(cid)
    if not cam: raise HTTPException(404)
    async def gen():
        last = None; idle = 0
        while not await request.is_disconnected():
            a = cam.annot
            if a is not None and a is not last:
                last = a; idle = 0; yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + a + b"\r\n"
            else:
                idle += 1
                if idle > 1500: break  # ~60 sn yeni kare yoksa kapat
            await asyncio.sleep(0.04)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")

SNAP = {"f": None}
@app.get("/api/snapshot")
def snapshot(cam: str = ""):
    c = CAMS.get(cam) or next((x for x in CAMS.values() if x.frame is not None), None)
    if c is None: raise HTTPException(400, "Kamera görüntüsü yok")
    with c.lock: f = None if c.frame is None else c.frame.copy()
    if f is None: raise HTTPException(400, "Kamera görüntüsü yok")
    SNAP["f"] = f.copy()
    for fc in E.detect(f):
        x, y, w, h = fc[:4].astype(int); cv2.rectangle(f, (x, y), (x + w, y + h), (220, 200, 0), 2)
    return Response(cv2.imencode(".jpg", f)[1].tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

# ---- kisiler
def _save_face(name, img, fc):
    d = PEOPLE / safe_name(name); d.mkdir(exist_ok=True)
    p = d / f"{int(time.time() * 1000)}.jpg"; imwrite_u(p, norm_crop(img, fc[4:14].reshape(5, 2))); return p.name

@app.post("/api/people")
def save_person(request: Request, d: dict):
    need(request, "mudur_yrd")
    name = safe_name(d.get("name", "")); old = d.get("old_name")
    if old and old != name and (PEOPLE / safe_name(old)).exists():
        if (PEOPLE / name).exists(): raise HTTPException(400, "Bu isimde kayıt zaten var")
        (PEOPLE / safe_name(old)).rename(PEOPLE / name); INFO_CACHE.pop(old, None)
        q("UPDATE events SET name=? WHERE name=?", (name, old))
    (PEOPLE / name).mkdir(exist_ok=True)
    info = {k: str(d.get(k, "")).strip() for k in ("no", "sinif", "veli_ad", "veli_tel", "not")}
    info["liste"] = d.get("liste") if d.get("liste") in ("ogrenci", "beyaz", "kara", "ziyaretci") else "ogrenci"
    (PEOPLE / name / "info.json").write_text(json.dumps(info, ensure_ascii=False), "utf-8")
    INFO_CACHE.pop(name, None)
    if old and old != name: train_bg()
    audit(request, "kişi kaydedildi", name, info["liste"] + (f" (eski ad: {old})" if old and old != name else "")); return {"ok": True}

@app.post("/api/enroll/capture")
def enroll_capture(request: Request, d: dict):
    need(request, "mudur_yrd")
    name = safe_name(d.get("name", "")); f = SNAP["f"]
    if f is None: raise HTTPException(400, "Kamera görüntüsü yok")
    faces = E.detect(f)
    if len(faces) == 0: raise HTTPException(400, "Karede yüz bulunamadı")
    return {"saved": _save_face(name, f, max(faces, key=lambda r: r[2] * r[3]))}

@app.post("/api/enroll/upload")
async def enroll_upload(request: Request, name: str = Form(...), files: list[UploadFile] = File(...)):
    need(request, "mudur_yrd"); name = safe_name(name); out = []
    for uf in files:
        img = cv2.imdecode(np.frombuffer(await uf.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None: out.append({"file": uf.filename, "error": "okunamadı"}); continue
        if max(img.shape[:2]) > 1600: s = 1600 / max(img.shape[:2]); img = cv2.resize(img, None, fx=s, fy=s)
        faces = E.detect(img)
        if len(faces) == 0: out.append({"file": uf.filename, "error": "yüz yok"}); continue
        out.append({"file": uf.filename, "saved": _save_face(name, img, max(faces, key=lambda r: r[2] * r[3]))})
    return out

@app.get("/api/people")
def people(request: Request):
    return [{"name": d.name, **read_info(d.name), "liste": liste_of(d.name),
             "photos": sorted(p.name for p in d.glob("*.jpg")), "trained": d.name in G.names}
            for d in sorted(PEOPLE.iterdir()) if d.is_dir() and visible(request, d.name)]

@app.get("/api/people/{name}/{photo}")
def photo(name: str, photo: str):
    if photo == "_":
        ps = sorted((PEOPLE / safe_name(name)).glob("*.jpg"))
        if not ps: raise HTTPException(404)
        p = ps[0]
    else: p = PEOPLE / safe_name(name) / safe_name(photo)
    if not p.exists(): raise HTTPException(404)
    return Response(p.read_bytes(), media_type="image/jpeg")

@app.delete("/api/people/{name}/{photo}")
def del_photo(request: Request, name: str, photo: str):
    need(request, "mudur_yrd"); (PEOPLE / safe_name(name) / safe_name(photo)).unlink(missing_ok=True); return {"ok": True}

@app.delete("/api/people/{name}")
def del_person(request: Request, name: str, purge: int = 0):
    """purge=1: mezun/ayrildi -> tum gecis ve mesaj kayitlari da silinir."""
    need(request, "mudur_yrd"); name = safe_name(name)
    shutil.rmtree(PEOPLE / name, ignore_errors=True); INFO_CACHE.pop(name, None)
    if purge:
        q("DELETE FROM events WHERE name=?", (name,)); q("DELETE FROM messages WHERE name=?", (name,))
        for r in q("SELECT id FROM pending WHERE cand=?", (name,)): drop_pending(r["id"])
    G.train(); audit(request, "kalıcı silindi (mezun/ayrıldı)" if purge else "kişi silindi", name); return {"ok": True}

@app.post("/api/train")
def train(request: Request): need(request, "mudur_yrd"); r = G.train(); audit(request, "model eğitildi", "", f"{len(r['people'])} kişi"); return r

# ---- gecisler / yoklama
@app.get("/api/events")
def events(request: Request, limit: int = 40, day: str = ""):
    rows = q("SELECT * FROM events WHERE time LIKE ? ORDER BY id DESC LIMIT ?", ((day or "") + "%", min(limit, 2000)))
    return [r for r in rows if visible(request, r["name"])]

# ---- yoklama
@app.get("/api/attendance")
def attendance(request: Request, day: str = ""):
    day = day or today_s(); st = {}
    for r in q("SELECT name,time,yon,gec FROM events WHERE time LIKE ? ORDER BY time", (day + "%",)):
        s = st.setdefault(r["name"], {"geldi": None, "gec": 0, "cikis": None, "son": None})
        if r["yon"] == "giris" and not s["geldi"]: s["geldi"] = r["time"][11:16]; s["gec"] = r["gec"]
        if r["yon"] == "cikis": s["cikis"] = r["time"][11:16]
        s["son"] = r["yon"]
    out = []
    for d in sorted(PEOPLE.iterdir()):
        if d.is_dir() and liste_of(d.name) == "ogrenci" and visible(request, d.name):
            s = st.get(d.name, {})
            durum = None if not s else ("cikti" if s.get("son") == "cikis" else "okulda")
            out.append({"name": d.name, **read_info(d.name), "geldi": s.get("geldi"), "gec": s.get("gec", 0),
                        "cikis": s.get("cikis"), "durum": durum})
    return out

@app.post("/api/attendance/manual")
def manual(request: Request, d: dict):
    name = safe_name(d.get("name", ""))
    if not visible(request, name): raise HTTPException(403, "Bu öğrenci senin sınıfında değil")
    log_event(name, 0, "elle:" + request.state.u["user"], "cikis" if d.get("yon") == "cikis" else "giris", kaynak="elle", force=True)
    audit(request, "elle " + ("çıkış" if d.get("yon") == "cikis" else "geldi"), name); return {"ok": True}

@app.post("/api/absent/notify")
def absent_notify(request: Request, d: dict):
    name = safe_name(d.get("name", ""))
    if not visible(request, name): raise HTTPException(403, "Bu öğrenci senin sınıfında değil")
    i = read_info(name)
    txt = f"Sayın {i.get('veli_ad') or 'Veli'}, öğrenciniz {name} bugün ({datetime.now():%d.%m.%Y}) okula gelmemiştir. Bilginize."
    audit(request, "gelmedi bildirildi", name)
    return {"wa": wa_send(i.get("veli_tel"), name, "gelmedi", txt, force_text=True)}

# ---- islem kaydi
def audit(request, action, target="", detail=""):
    u = getattr(getattr(request, "state", None), "u", None) or {"user": "?"}
    ins("INSERT INTO audit(time,user,action,target,detail) VALUES(?,?,?,?,?)", (now_s(), u["user"], action, target, str(detail)[:300]))

@app.get("/api/audit")
def audit_list(request: Request, limit: int = 500, user: str = "", qs: str = ""):
    need(request, "yonetici")
    return q("SELECT * FROM audit WHERE user LIKE ? AND (action LIKE ? OR target LIKE ?) ORDER BY id DESC LIMIT ?",
             (user or "%", f"%{qs}%", f"%{qs}%", min(limit, 5000)))

# ---- raporlar
@app.get("/api/reports")
def reports(request: Request, start: str = "", end: str = "", month: str = "", sinif: str = "", name: str = "", t1: str = "", t2: str = ""):
    """Tarih araligi (start..end), opsiyonel sinif / ogrenci / saat araligi (t1..t2, giris saati) ile devam raporu."""
    if month and not start:
        y, m = map(int, month.split("-")); start = f"{month}-01"
        end = (datetime(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)).strftime("%Y-%m-%d")
    start = start or datetime.now().strftime("%Y-%m-01"); end = min(end or today_s(), today_s())
    d0, d1 = datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")
    if (d1 - d0).days > 400: raise HTTPException(400, "En fazla 400 günlük aralık seçilebilir")
    days = [(d0 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((d1 - d0).days + 1)] if d1 >= d0 else []
    school = [d for d in days if not is_off_day(d)]
    # her ogrenci/gun icin ilk giris ve son cikis
    rows = q("""SELECT name, substr(time,1,10) day, MIN(CASE WHEN yon='giris' THEN time END) gir, MAX(CASE WHEN yon='cikis' THEN time END) cik
                FROM events WHERE time >= ? AND time < ? AND liste='ogrenci' GROUP BY name, day""", (start, (d1 + timedelta(days=1)).strftime("%Y-%m-%d")))
    def in_t(t): return t and (not t1 or t[11:16] >= t1) and (not t2 or t[11:16] <= t2)
    studs_all = [d.name for d in sorted(PEOPLE.iterdir()) if d.is_dir() and liste_of(d.name) == "ogrenci" and visible(request, d.name)]
    studs_f = [n for n in studs_all if (not sinif or read_info(n).get("sinif") == sinif) and (not name or n == name)]
    sset = set(studs_f); sch = set(school)
    pres, late, daily, detail = Counter(), Counter(), Counter(), {}
    for r in rows:
        if r["name"] not in sset or r["day"] not in sch or not in_t(r["gir"]): continue
        pres[r["name"]] += 1; daily[r["day"]] += 1
        g = is_late(r["gir"]); late[r["name"]] += g
        if name: detail[r["day"]] = {"giris": r["gir"][11:16], "cikis": (r["cik"] or "")[11:16], "gec": g}
    studs = []
    for n in studs_f:
        i = read_info(n); p = pres[n]
        studs.append({"name": n, "no": i.get("no", ""), "sinif": i.get("sinif", ""), "gelen": p, "gec": late[n],
                      "gun": len(school), "gelmedi": len(school) - p, "oran": round(100 * p / len(school), 1) if school else 0})
    cls = {}
    for s in studs:
        c = cls.setdefault(s["sinif"] or "—", {"sinif": s["sinif"] or "—", "ogrenci": 0, "oran_top": 0, "gec": 0})
        c["ogrenci"] += 1; c["oran_top"] += s["oran"]; c["gec"] += s["gec"]
    out = {"start": start, "end": end, "school_days": len(school), "students": studs,
           "classes": sorted(({"sinif": c["sinif"], "ogrenci": c["ogrenci"], "oran": round(c["oran_top"] / c["ogrenci"], 1), "gec": c["gec"]} for c in cls.values()), key=lambda c: c["sinif"]),
           "daily": [{"day": d, "gelen": daily[d], "toplam": len(studs_f)} for d in school],
           "all_classes": sorted({read_info(n).get("sinif") for n in studs_all if read_info(n).get("sinif")}),
           "all_students": [{"name": n, "sinif": read_info(n).get("sinif", "")} for n in studs_all]}
    if name:
        out["detail"] = [{"day": d, "off": is_off_day(d), **detail.get(d, {})} for d in days]
    return out

@app.get("/api/attendance/range")
def attendance_range(request: Request, start: str, end: str, sinif: str = "", name: str = "", t1: str = "", t2: str = "", durum: str = ""):
    """Iki tarih arasi gun gun yoklama: her okul gunu x ogrenci icin giris/cikis/gec/gelmedi."""
    d0, d1 = datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(min(end, today_s()), "%Y-%m-%d")
    if (d1 - d0).days > 92: raise HTTPException(400, "Yoklama için en fazla 92 gün seçilebilir (daha uzunu için Raporlar)")
    school = [(d0 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((d1 - d0).days + 1)]
    school = [d for d in school if not is_off_day(d)]
    ev = {}
    for r in q("""SELECT name, substr(time,1,10) day, MIN(CASE WHEN yon='giris' THEN time END) gir, MAX(CASE WHEN yon='cikis' THEN time END) cik
                  FROM events WHERE time >= ? AND time < ? AND liste='ogrenci' GROUP BY name, day""",
               (start, (d1 + timedelta(days=1)).strftime("%Y-%m-%d"))):
        ev[(r["name"], r["day"])] = r
    studs = [d.name for d in sorted(PEOPLE.iterdir()) if d.is_dir() and liste_of(d.name) == "ogrenci" and visible(request, d.name)
             and (not sinif or read_info(d.name).get("sinif") == sinif) and (not name or d.name == name)]
    out = []
    for day in reversed(school):
        for n in studs:
            r = ev.get((n, day)); gir = r and r["gir"]
            if gir and ((t1 and gir[11:16] < t1) or (t2 and gir[11:16] > t2)): continue
            g = bool(gir and is_late(gir))
            st = "gec" if g else "geldi" if gir else "gelmedi"
            if durum and durum != st: continue
            i = read_info(n)
            out.append({"day": day, "name": n, "no": i.get("no", ""), "sinif": i.get("sinif", ""), "giris": (gir or "")[11:16],
                        "cikis": ((r and r["cik"]) or "")[11:16], "durum": st, "veli_ad": i.get("veli_ad", ""), "veli_tel": i.get("veli_tel", "")})
            if len(out) >= 20000: return {"rows": out, "truncated": True, "days": len(school)}
    return {"rows": out, "truncated": False, "days": len(school)}

# ---- toplu duyuru
@app.post("/api/broadcast")
def broadcast(request: Request, d: dict):
    text = (d.get("text") or "").strip()
    if len(text) < 3: raise HTTPException(400, "Mesaj boş")
    if not wa_ready(): raise HTTPException(400, "WhatsApp ayarları yapılmamış")
    sinif = (d.get("sinif") or "").strip()
    mc = my_class(request)
    if mc: sinif = mc
    elif not sinif: need(request, "mudur_yrd")
    hedef = []
    for p in sorted(PEOPLE.iterdir()):
        if not p.is_dir() or liste_of(p.name) != "ogrenci": continue
        i = read_info(p.name)
        if sinif and i.get("sinif") != sinif: continue
        if wa_phone(i.get("veli_tel")): hedef.append((p.name, i))
    if d.get("dry"): return {"count": len(hedef)}
    def run():
        for name, i in hedef:
            try: msg = text.format(veli=i.get("veli_ad") or "Veli", ogrenci=name)
            except Exception: msg = text
            wa_send(i.get("veli_tel"), name, "duyuru", msg, force_text=True); time.sleep(1.5)
    threading.Thread(target=run, daemon=True).start()
    audit(request, "toplu duyuru", sinif or "tüm okul", f"{len(hedef)} veli: {text[:120]}")
    return {"count": len(hedef)}

# ---- veli bot (whapi gelen mesaj)
def bot_reply(tel, text):
    if not S.get("bot_enabled"): return
    kw = [k.strip().lower() for k in (S.get("bot_keyword") or "durum").split(",") if k.strip()]
    if not any(k in text.lower() for k in kw): return
    kids = [p.name for p in PEOPLE.iterdir() if p.is_dir() and wa_phone(read_info(p.name).get("veli_tel")) == tel]
    if not kids:
        wa_send(tel, "?", "bot", "Bu numara sistemde bir öğrenci velisi olarak kayıtlı değil. Lütfen okul ile iletişime geçin.", force_text=True); return
    lines = []
    for k in kids:
        ev = q("SELECT time,yon FROM events WHERE name=? AND time LIKE ? ORDER BY time", (k, today_s() + "%"))
        gir = next((e["time"][11:16] for e in ev if e["yon"] == "giris"), None)
        cik = next((e["time"][11:16] for e in reversed(ev) if e["yon"] == "cikis"), None)
        if cik: lines.append(f"• {k}: {gir or '?'} giriş, {cik} çıkış yaptı.")
        elif gir: lines.append(f"• {k}: bugün {gir}'de okula giriş yaptı, şu an okulda.")
        else: lines.append(f"• {k}: bugün henüz giriş kaydı yok.")
    wa_send(tel, ", ".join(kids), "bot", f"Bugün ({datetime.now():%d.%m.%Y}):\n" + "\n".join(lines), force_text=True)

# ---- excel ile toplu kayit
HDR = {"ad soyad": "name", "adsoyad": "name", "ad": "name", "isim": "name", "ogrenci": "name", "no": "no", "okul no": "no",
       "numara": "no", "sinif": "sinif", "sube": "sinif", "veli": "veli_ad", "veli ad": "veli_ad", "veli adi": "veli_ad",
       "veli ad soyad": "veli_ad", "veli tel": "veli_tel", "veli telefon": "veli_tel", "telefon": "veli_tel", "tel": "veli_tel",
       "gsm": "veli_tel", "not": "not"}
def norm_h(s): return re.sub(r"\s+", " ", ascii_name(str(s or "")).lower().replace(".", "").replace("_", " ")).strip()

@app.get("/api/import/template")
def import_template(request: Request):
    import openpyxl
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Ogrenciler"
    ws.append(["Ad Soyad", "No", "Sınıf", "Veli Ad Soyad", "Veli Tel"])
    ws.append(["Ahmet Yılmaz", "101", "5-A", "Ayşe Yılmaz", "05321112233"])
    for col, w in zip("ABCDE", (24, 8, 8, 24, 16)): ws.column_dimensions[col].width = w
    b = io.BytesIO(); wb.save(b)
    return Response(b.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": 'attachment; filename="ogrenci-sablon.xlsx"'})

@app.post("/api/import/excel")
async def import_excel(request: Request, file: UploadFile = File(...)):
    need(request, "mudur_yrd")
    import openpyxl
    try: ws = openpyxl.load_workbook(io.BytesIO(await file.read()), read_only=True, data_only=True).active
    except Exception: raise HTTPException(400, "Excel dosyası okunamadı (.xlsx olmalı)")
    rows = list(ws.iter_rows(values_only=True))
    if not rows: raise HTTPException(400, "Dosya boş")
    cols = {i: HDR.get(norm_h(h)) for i, h in enumerate(rows[0])}
    if "name" not in cols.values(): raise HTTPException(400, "'Ad Soyad' sütunu bulunamadı")
    new = upd = 0; errs = []
    for n, r in enumerate(rows[1:], start=2):
        d = {cols[i]: ("" if v is None else str(v).strip()) for i, v in enumerate(r) if cols.get(i)}
        if d.get("no", "").endswith(".0"): d["no"] = d["no"][:-2]
        if d.get("veli_tel") and d["veli_tel"].replace(".0", "").isdigit() and len(d["veli_tel"].replace(".0", "")) == 10:
            d["veli_tel"] = "0" + d["veli_tel"].replace(".0", "")
        try: name = safe_name(d.get("name", ""))
        except HTTPException: errs.append(f"Satır {n}: isim boş"); continue
        p = PEOPLE / name; ex = p.exists(); p.mkdir(exist_ok=True)
        info = read_info(name) if ex else {"liste": "ogrenci"}
        info.update({k: d[k] for k in ("no", "sinif", "veli_ad", "veli_tel", "not") if d.get(k)})
        (p / "info.json").write_text(json.dumps(info, ensure_ascii=False), "utf-8"); INFO_CACHE.pop(name, None)
        upd += ex; new += not ex
    audit(request, "excel içe aktarma", file.filename, f"{new} yeni, {upd} güncellendi")
    return {"new": new, "updated": upd, "errors": errs[:50]}

@app.post("/api/import/photos")
async def import_photos(request: Request, file: UploadFile = File(...)):
    """Zip: okulno.jpg / okulno_2.jpg / Ad Soyad.jpg / KlasorAdi(okulno)/x.jpg"""
    need(request, "mudur_yrd")
    try: z = zipfile.ZipFile(io.BytesIO(await file.read()))
    except Exception: raise HTTPException(400, "Zip dosyası okunamadı")
    by_no = {}; by_name = {}
    for p in PEOPLE.iterdir():
        if p.is_dir():
            i = read_info(p.name)
            if i.get("no"): by_no[str(i["no"]).strip()] = p.name
            by_name[norm_h(p.name)] = p.name
    ok = 0; miss = []; noface = []
    for zn in z.namelist():
        if zn.endswith("/") or not re.search(r"\.(jpe?g|png|bmp|webp)$", zn, re.I): continue
        parts = zn.replace("\\", "/").split("/")
        key = parts[-2] if len(parts) > 1 else re.sub(r"[_\-\s]\d{1,2}$", "", Path(parts[-1]).stem)
        stem = Path(parts[-1]).stem
        name = by_no.get(key.strip()) or by_name.get(norm_h(key)) or by_no.get(stem) or by_name.get(norm_h(stem))
        if not name: miss.append(zn); continue
        img = cv2.imdecode(np.frombuffer(z.read(zn), np.uint8), cv2.IMREAD_COLOR)
        if img is None: miss.append(zn); continue
        if max(img.shape[:2]) > 1600: s = 1600 / max(img.shape[:2]); img = cv2.resize(img, None, fx=s, fy=s)
        faces = E.detect(img)
        if len(faces) == 0: noface.append(zn); continue
        _save_face(name, img, max(faces, key=lambda r: r[2] * r[3])); ok += 1
    rep = G.train()
    audit(request, "toplu foto yükleme", file.filename, f"{ok} foto eklendi")
    return {"saved": ok, "unmatched": miss[:100], "noface": noface[:100], "trained": len(rep["people"])}

# ---- onay bekleyen / bilinmeyen
@app.get("/api/pending")
def pending(request: Request, kind: str = "onay"):
    rows = q("SELECT id,time,kind,cand,score,img,cam,grp FROM pending WHERE kind=? ORDER BY id DESC LIMIT 300", (kind,))
    return [r for r in rows if kind != "onay" or visible(request, r["cand"])]

@app.get("/api/unknown/groups")
def unknown_groups(request: Request):
    rows = q("SELECT id,time,score,img,cam,grp,emb FROM pending WHERE kind='bilinmeyen' ORDER BY id DESC")
    gs = {}
    for r in rows:
        g = gs.setdefault(r["grp"] or "?", {"id": r["grp"] or "?", "n": 0, "items": [], "first": r["time"], "last": r["time"], "cams": set(), "e": []})
        g["n"] += 1; g["first"] = min(g["first"], r["time"]); g["last"] = max(g["last"], r["time"]); g["cams"].add(r["cam"] or "")
        g["items"].append({"id": r["id"], "img": r["img"], "time": r["time"], "score": r["score"]})
        if r["emb"]: g["e"].append(np.frombuffer(r["emb"], np.float32))
    out = []
    for g in gs.values():
        best = None
        if g["e"]:
            c = np.mean(g["e"], 0); c /= np.linalg.norm(c)
            n_, s_ = G.match(c[None])[0]
            if n_: best = {"name": n_, "score": round(s_, 3)}
            # gruba en cok benzeyen (merkeze yakin) yuzler once
            sims = np.stack(g["e"]) @ c
            if len(sims) == len(g["items"]): g["items"] = [g["items"][i] for i in np.argsort(-sims)]
        out.append({"id": g["id"], "n": g["n"], "first": g["first"], "last": g["last"], "cams": sorted(g["cams"]),
                    "items": g["items"][:40], "best": best})
    out.sort(key=lambda g: (-g["n"], g["last"]), reverse=False)
    return {"groups": out, "threshold": S.get("cluster_threshold", 0.45)}

@app.post("/api/unknown/groups/{gid}/assign")
def unknown_assign(request: Request, gid: str, d: dict):
    """Grubu bir kisiye ata: en iyi kareler galeriye eklenir, grup silinir."""
    need(request, "mudur_yrd")
    name = safe_name(d.get("name", ""))
    if not (PEOPLE / name).exists():
        (PEOPLE / name).mkdir()
        info = {"liste": d.get("liste") if d.get("liste") in ("ogrenci", "beyaz", "kara") else "ogrenci",
                "sinif": d.get("sinif", ""), "not": d.get("not", "Bilinmeyen yüz grubundan eklendi")}
        (PEOPLE / name / "info.json").write_text(json.dumps(info, ensure_ascii=False), "utf-8"); INFO_CACHE.pop(name, None)
    rows = q("SELECT id, emb FROM pending WHERE kind='bilinmeyen' AND grp=?", (gid,))
    if not rows: raise HTTPException(404, "Grup bulunamadı")
    # en farkli 8 kareyi sec
    E_ = [np.frombuffer(r["emb"], np.float32) if r["emb"] else None for r in rows]
    idx = [i for i, e in enumerate(E_) if e is not None]
    pick = idx[:1]
    if idx:
        X = np.stack([E_[i] for i in idx])
        sel = [0]
        while len(sel) < min(int(d.get("learn", 8)), len(idx)):
            dd = 1 - (X @ X[sel].T).max(1); dd[sel] = -1; sel.append(int(dd.argmax()))
        pick = [idx[k] for k in sel]
    for k in pick:
        c = UNKNOWN / f"{rows[k]['id']}_crop.jpg"
        if c.exists(): shutil.copy(c, PEOPLE / name / f"{int(time.time() * 1000)}_{k}.jpg")
    for r in rows: drop_pending(r["id"])
    train_bg(); audit(request, "bilinmeyen grup atandı", name, f"{gid}: {len(rows)} görülme, {len(pick)} foto")
    return {"ok": True, "photos": len(pick), "n": len(rows)}

@app.delete("/api/unknown/groups/{gid}")
def unknown_del_group(request: Request, gid: str):
    for r in q("SELECT id FROM pending WHERE kind='bilinmeyen' AND grp=?", (gid,)): drop_pending(r["id"])
    audit(request, "bilinmeyen grup silindi", gid); return {"ok": True}

@app.post("/api/unknown/move")
def unknown_move(request: Request, d: dict):
    """Tek yuzu baska gruba tasi ya da yeni gruba ayir; iki grubu birlestir."""
    if d.get("merge"):
        a, b = d["merge"]; q("UPDATE pending SET grp=? WHERE grp=? AND kind='bilinmeyen'", (a, b)); return {"ok": True}
    q("UPDATE pending SET grp=? WHERE id=?", (d.get("to") or new_gid(), int(d["id"]))); return {"ok": True}

@app.post("/api/unknown/recluster")
def unknown_recluster(request: Request, d: dict):
    need(request, "mudur_yrd")
    if d.get("threshold"): S["cluster_threshold"] = float(d["threshold"]); save_settings()
    return {"groups": recluster()}

@app.get("/media/snap/{f}")
def media_snap(f: str):
    p = SNAPS / safe_name(f)
    if not p.exists(): raise HTTPException(404)
    return Response(p.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

@app.get("/api/events/{eid}")
def event_detail(request: Request, eid: int):
    e = q("SELECT * FROM events WHERE id=?", (eid,), one=True)
    if not e or not visible(request, e["name"]): raise HTTPException(404)
    p = PEOPLE / e["name"]
    e["photos"] = sorted(x.name for x in p.glob("*.jpg")) if p.exists() else []
    e["info"] = read_info(e["name"]); return e

@app.post("/api/events/{eid}/verify")
def event_verify(request: Request, eid: int, d: dict):
    """Operator kontrolu: dogru / yanlis. Yanlissa gecis silinir, yuz onaya duser; istenirse dogru kisiye yazilir."""
    e = q("SELECT * FROM events WHERE id=?", (eid,), one=True)
    if not e or not visible(request, e["name"]): raise HTTPException(404)
    u = request.state.u["user"]
    if d.get("ok"):
        q("UPDATE events SET dogrulama=? WHERE id=?", ("dogru:" + u, eid)); audit(request, "geçiş doğrulandı", e["name"], f"#{eid}"); return {"ok": True}
    q("DELETE FROM events WHERE id=?", (eid,))
    fp = SNAPS / f"{eid}_f.jpg"; fr = SNAPS / f"{eid}.jpg"
    dogru = (d.get("dogru") or "").strip()
    if dogru:
        dogru = safe_name(dogru)
        if not (PEOPLE / dogru).exists(): raise HTTPException(400, "Doğru kişi bulunamadı")
        log_event(dogru, e["score"] or 0, e["cam"], e["yon"] or "giris", kaynak="düzeltme:" + u, force=True)
    elif fp.exists():
        img = imread_u(fp); faces = E.detect(img) if img is not None else []
        if len(faces):
            fc = max(faces, key=lambda r: r[2] * r[3]); add_pending("onay", e["name"], e["score"] or 0, img, norm_crop(img, fc[4:14].reshape(5, 2)), e["cam"])
    for x in (fp, fr): x.unlink(missing_ok=True)
    audit(request, "yanlış geçiş silindi", e["name"], f"#{eid} {e['time']}" + (f" → {dogru}" if dogru else ""))
    return {"ok": True}

@app.get("/media/{kind}/{f}")
def media(kind: str, f: str):
    p = (PENDING if kind == "onay" else UNKNOWN) / safe_name(f)
    if not p.exists(): raise HTTPException(404)
    return Response(p.read_bytes(), media_type="image/jpeg")

@app.post("/api/pending/{pid}/approve")
def approve(request: Request, pid: int, d: dict):
    r = q("SELECT * FROM pending WHERE id=?", (pid,), one=True)
    if not r: raise HTTPException(404)
    name = safe_name(d.get("name") or r["cand"])
    if not visible(request, name): raise HTTPException(403, "Bu öğrenci senin sınıfında değil")
    if not (PEOPLE / name).exists():
        need(request, "mudur_yrd"); (PEOPLE / name).mkdir()
        (PEOPLE / name / "info.json").write_text(json.dumps({"liste": "ogrenci", "sinif": d.get("sinif", "")}, ensure_ascii=False), "utf-8")
    folder = PENDING if r["kind"] == "onay" else UNKNOWN
    crop = folder / f"{pid}_crop.jpg"
    if crop.exists() and d.get("learn", True): shutil.copy(crop, PEOPLE / name / f"{int(time.time() * 1000)}.jpg")
    if d.get("mark", True): log_event(name, r["score"], r["cam"], kaynak="onay:" + request.state.u["user"], force=True)
    drop_pending(pid); train_bg(); audit(request, "onaylandı", name, f"{r['kind']} #{pid} skor {r['score']}"); return {"ok": True}

@app.delete("/api/pending/{pid}")
def reject(request: Request, pid: int): drop_pending(pid); audit(request, "onay reddedildi", f"#{pid}"); return {"ok": True}

@app.delete("/api/pending")
def clear_pending(request: Request, kind: str = "bilinmeyen"):
    need(request, "mudur_yrd")
    for r in q("SELECT id FROM pending WHERE kind=?", (kind,)): drop_pending(r["id"])
    return {"ok": True}

# ---- mesajlar
@app.get("/api/messages")
def messages(request: Request, limit: int = 200):
    rows = q("SELECT * FROM messages ORDER BY id DESC LIMIT ?", (min(limit, 2000),))
    return [r for r in rows if r["kind"] == "test" or visible(request, r["name"])]

@app.get("/api/wa_link/{name}")
def wa_link(request: Request, name: str):
    i = read_info(safe_name(name)); t = now_s()
    return {"url": f"https://wa.me/{wa_phone(i.get('veli_tel'))}?text=" + urllib.parse.quote(S["wa_template"].format(**wa_vars(i, name, t)))}

@app.post("/api/wa_test")
def wa_test(request: Request, d: dict):
    need(request, "yonetici")
    if not wa_ready(): raise HTTPException(400, "Önce WhatsApp token bilgilerini kaydet")
    tel = d.get("tel", "")
    if S["wa_provider"] == "meta" and S["wa_mode"] == "template" and S["wa_tpl_name"]:
        st = wa_send(tel, "TEST", "test", {"veli": "Test Veli", "ogrenci": "Test Öğrenci", "saat": datetime.now().strftime("%H:%M")}, force_text=True)
    else:
        st = wa_send(tel, "TEST", "test", "Hype Vision test mesajı ✓", force_text=True)
    if st == "numara yok": raise HTTPException(400, "Telefon numarası geçersiz")
    time.sleep(3)
    m = q("SELECT * FROM messages WHERE kind='test' ORDER BY id DESC LIMIT 1", one=True)
    return {"status": m["status"] if m else st, "err": (m or {}).get("err")}

# Meta webhook: iletildi / okundu durumlari (sunucu internetten erisilebilir olmali)
@app.get("/webhook/whatsapp")
def wh_verify(request: Request):
    p = request.query_params
    if p.get("hub.verify_token") == S["wa_verify"]: return PlainTextResponse(p.get("hub.challenge", ""))
    raise HTTPException(403)

@app.post("/webhook/whatsapp")
async def wh(request: Request):
    body = await request.json()
    tr = {"sent": "gonderildi", "delivered": "iletildi", "read": "okundu", "failed": "hata", "played": "okundu"}
    for m in body.get("messages", []):  # whapi gelen mesaj -> veli bot
        if m.get("from_me"): continue
        txt = (m.get("text") or {}).get("body") or ""
        tel = wa_phone(m.get("from") or (m.get("chat_id") or "").split("@")[0])
        if txt and tel: threading.Thread(target=bot_reply, args=(tel, txt), daemon=True).start()
    for s in body.get("statuses", []):  # whapi
        q("UPDATE messages SET status=? WHERE wamid=?", (tr.get(s.get("status"), s.get("status")), s.get("id")))
    for e in body.get("entry", []):     # meta
        for c in e.get("changes", []):
            for s in c.get("value", {}).get("statuses", []):
                q("UPDATE messages SET status=? WHERE wamid=?", (tr.get(s.get("status"), s.get("status")), s.get("id")))
    return {"ok": True}

# ---- yedek
@app.get("/api/backup")
def backup(request: Request):
    need(request, "yonetici")
    buf = io.BytesIO()
    with DBL: CONN.commit()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in PEOPLE.rglob("*"):
            if p.is_file(): z.write(p, "people/" + p.relative_to(PEOPLE).as_posix())
        for p in (GALLERY, SETTINGS, DB_PATH):
            if p.exists(): z.write(p, p.name)
    fn = f"hypevision-yedek-{datetime.now():%Y%m%d-%H%M}.zip"
    return Response(buf.getvalue(), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{fn}"'})

@app.post("/api/restore")
async def restore(request: Request, file: UploadFile = File(...)):
    need(request, "yonetici")
    global CONN
    data = await file.read()
    try: z = zipfile.ZipFile(io.BytesIO(data))
    except Exception: raise HTTPException(400, "Geçerli bir yedek zip dosyası değil")
    names = z.namelist()
    if not any(n.startswith("people/") for n in names): raise HTTPException(400, "Yedekte öğrenci verisi yok")
    for c in CAMS.values(): c.stop()
    shutil.rmtree(PEOPLE, ignore_errors=True); PEOPLE.mkdir()
    for n in names:
        if n.startswith("people/") and not n.endswith("/") and ".." not in n:
            t = PEOPLE / n[7:]; t.parent.mkdir(parents=True, exist_ok=True); t.write_bytes(z.read(n))
    if "settings.json" in names: SETTINGS.write_bytes(z.read("settings.json")); S.clear(); S.update(load_settings())
    if "okul.db" in names:
        with DBL:
            CONN.close(); DB_PATH.write_bytes(z.read("okul.db")); CONN = db()
    INFO_CACHE.clear(); G.train(); apply_cameras(); audit(request, "yedekten geri yükleme", file.filename)
    return {"ok": True, "people": len(G.names)}

# ---- video: kayit ve analiz
UPLOADS = DATA / "uploads"; UPLOADS.mkdir(exist_ok=True)
VENROLL = {}   # token -> {"name","crops":[...], "t": time}
VJOBS = {}     # job id -> durum

async def save_upload(file: UploadFile):
    ext = (Path(file.filename or "v.mp4").suffix or ".mp4").lower()
    if ext not in (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".ts", ".webm", ".dav", ".h264"):
        raise HTTPException(400, "Desteklenmeyen video türü")
    p = UPLOADS / f"{secrets.token_hex(8)}{ext}"
    with open(p, "wb") as fh:
        while chunk := await file.read(1 << 20): fh.write(chunk)
    return p

def b64jpg(img, q=88): return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()

def frontalness(kps):
    """0..1: burun gozlerin ortasina ne kadar yakinsa o kadar onden."""
    le, re_, nose = kps[0], kps[1], kps[2]
    ed = np.linalg.norm(re_ - le) + 1e-6
    return float(max(0.0, 1 - abs(nose[0] - (le[0] + re_[0]) / 2) / ed * 2))

@app.post("/api/video/enroll")
async def video_enroll(request: Request, name: str = Form(...), count: int = Form(5), file: UploadFile = File(...)):
    """Videodan en net ve birbirinden farkli kareleri sec (kaydetmeden onizleme doner)."""
    need(request, "mudur_yrd"); name = safe_name(name)
    p = await save_upload(file)
    try:
        cap = cv2.VideoCapture(str(p)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, n // 120) if n else 3
        cands, i = [], 0
        while True:
            ok = cap.grab()
            if not ok: break
            if i % step == 0:
                ok, f = cap.retrieve()
                if ok:
                    sc = 1.0; img = f
                    if f.shape[1] > 1280: sc = 1280 / f.shape[1]; img = cv2.resize(f, None, fx=sc, fy=sc)
                    faces = E.detect(img)
                    if len(faces):
                        fc = max(faces, key=lambda r: r[2] * r[3]).copy(); fc[:14] /= sc
                        if min(fc[2], fc[3]) >= 40:
                            crop = norm_crop(f, fc[4:14].reshape(5, 2))
                            sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                            front = frontalness(fc[4:14].reshape(5, 2))
                            q_ = min(sharp / 150, 1.5) * (0.4 + 0.6 * front) * min(fc[2] / 120, 1.2) * float(fc[14])
                            cands.append({"crop": crop, "q": q_, "t": i / (cap.get(cv2.CAP_PROP_FPS) or 25), "front": front, "size": int(fc[2])})
            i += 1
        cap.release()
    finally: p.unlink(missing_ok=True)
    if not cands: raise HTTPException(400, "Videoda yüz bulunamadı")
    cands.sort(key=lambda c: -c["q"]); cands = cands[:max(12, len(cands) // 2)]  # kaliteli yari
    emb = E.embed([c["crop"] for c in cands])
    # cesitlilik: en kaliteliyi al, sonra secilenlere en uzak olani ekle (kaliteyle agirlikli)
    pick = [0]
    while len(pick) < min(count, len(cands)):
        d = 1 - (emb @ emb[pick].T).max(1)
        sc = d * (0.5 + np.array([c["q"] for c in cands]) / max(c["q"] for c in cands))
        sc[pick] = -1; pick.append(int(sc.argmax()))
    # galeride baska biriyle karisiyor mu?
    warn = None
    m = G.match(emb[pick])
    others = [(n_, s) for n_, s in m if n_ and n_ != name and s >= S["threshold"]]
    if others: warn = f"Dikkat: bu kareler galerideki '{others[0][0]}' kişisine de benziyor ({others[0][1]:.2f}). Doğru kişi olduğundan emin ol."
    tok = secrets.token_hex(8)
    VENROLL[tok] = {"name": name, "crops": [cands[k]["crop"] for k in pick], "t": time.time()}
    for k in [k for k, v in VENROLL.items() if time.time() - v["t"] > 1800]: VENROLL.pop(k, None)
    return {"token": tok, "frames": len(cands), "warn": warn,
            "picks": [{"i": j, "img": b64jpg(cands[k]["crop"]), "t": round(cands[k]["t"], 1), "front": round(cands[k]["front"], 2),
                       "size": cands[k]["size"]} for j, k in enumerate(pick)]}

@app.post("/api/video/enroll/save")
def video_enroll_save(request: Request, d: dict):
    need(request, "mudur_yrd")
    v = VENROLL.pop(d.get("token", ""), None)
    if not v: raise HTTPException(400, "Önizleme süresi doldu, videoyu tekrar yükle")
    d_ = PEOPLE / v["name"]; d_.mkdir(exist_ok=True)
    idx = [i for i in d.get("idx", []) if 0 <= i < len(v["crops"])]
    for i in idx: imwrite_u(d_ / f"{int(time.time() * 1000)}_{i}.jpg", v["crops"][i])
    train_bg(); audit(request, "videodan kayıt", v["name"], f"{len(idx)} kare")
    return {"saved": len(idx)}

def _analyze(jid, path, every):
    J = VJOBS[jid]
    try:
        cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
        J["total"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); tracks = []; i = 0
        VG = {"id": [], "c": [], "n": []}  # bu videodaki bilinmeyen kisi gruplari
        def close(tr):
            if len(tr.votes) < 2 and not tr.done: return
            sc, nm, thumb, _ = tr.best
            if tr.done: kind = "tanindi"
            elif sc >= S["low_threshold"]: kind = "emin_degil"
            else: kind = "bilinmeyen"
            grp = None
            if kind != "tanindi" and tr.best[3] is not None:
                e = E.embed([tr.best[3]])[0]
                if VG["c"]:
                    sim = np.stack(VG["c"]) @ e; k = int(sim.argmax())
                    if sim[k] >= S.get("cluster_threshold", 0.45):
                        grp = VG["id"][k]; n_ = VG["n"][k]; m = VG["c"][k] * n_ + e; VG["c"][k] = m / np.linalg.norm(m); VG["n"][k] += 1
                if grp is None:
                    grp = f"#{len(VG['id']) + 1}"; VG["id"].append(grp); VG["c"].append(e); VG["n"].append(1)
            J["results"].append({"t0": round(tr.t0, 1), "t1": round(tr.last, 1), "name": tr.decided or nm, "score": round(tr.mean if tr.done else sc, 3),
                                 "kind": kind, "liste": liste_of(tr.decided) if tr.decided else None, "frames": len(tr.votes), "grp": grp,
                                 "img": b64jpg(thumb, 80) if thumb is not None else None})
        while not J.get("cancel"):
            ok = cap.grab()
            if not ok: break
            if i % every == 0:
                ok, f = cap.retrieve()
                if not ok: break
                ts = i / fps
                dw = int(S["det_width"]); scale = 1.0; img = f
                if f.shape[1] > dw: scale = dw / f.shape[1]; img = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                used, need_ = set(), []
                for fc in E.detect(img):
                    fc = fc.copy(); fc[:14] /= scale; box = tuple(fc[:4])
                    if min(fc[2], fc[3]) < S["min_face"]: continue
                    tr = max((t for t in tracks if id(t) not in used), key=lambda t: iou(t.box, box), default=None)
                    if tr is None or iou(tr.box, box) < 0.2: tr = Track(box); tr.t0 = ts; tracks.append(tr)
                    used.add(id(tr)); tr.box = box; tr.last = ts
                    if not tr.done: need_.append((fc, tr))
                crops = [norm_crop(f, fc[4:14].reshape(5, 2)) for fc, _ in need_]
                for (fc, tr), crop, m in zip(need_, crops, G.match_full(E.embed(crops))):
                    acc, why = decide(m, face_quality(crop, fc)); nm, sc = m["name"], m["score"]
                    if why == "kalite": continue
                    tr.votes.append((acc, sc)); tr.votes = tr.votes[-15:]
                    if sc > tr.best[0]: tr.best = (sc, nm, face_thumb(f, fc), crop)
                    top = vote_result(tr)
                    if top: tr.done = True; tr.decided = top; tr.mean = float(np.mean([v[1] for v in tr.votes if v[0] == top]))
                for tr in [t for t in tracks if ts - t.last > 1.5]: close(tr); tracks.remove(tr)
                J["preview"] = None
            i += 1; J["done"] = i
        for tr in tracks: close(tr)
        cap.release()
        if J["done"] == 0: J["status"] = "hata"; J["err"] = "Video açılamadı ya da bozuk (desteklenen: mp4, avi, mov, mkv, ts)"
        else: J["status"] = "iptal" if J.get("cancel") else "bitti"
    except Exception as e: J["status"] = "hata"; J["err"] = str(e)[:200]
    finally:
        Path(path).unlink(missing_ok=True); J["results"].sort(key=lambda r: r["t0"]); J["end"] = time.time()

@app.post("/api/video/analyze")
async def video_analyze(request: Request, every: int = Form(2), file: UploadFile = File(...)):
    """Kayitli videoyu analiz et. Yoklamaya kayit dusmez, mesaj gitmez."""
    need(request, "mudur_yrd")
    p = await save_upload(file); jid = secrets.token_hex(6)
    VJOBS[jid] = {"id": jid, "file": file.filename, "status": "isleniyor", "done": 0, "total": 0, "results": [], "start": time.time(), "every": every}
    threading.Thread(target=_analyze, args=(jid, p, max(1, min(every, 10))), daemon=True).start()
    audit(request, "video analiz", file.filename)
    return {"id": jid}

@app.get("/api/video/job/{jid}")
def video_job(jid: str):
    J = VJOBS.get(jid)
    if not J: raise HTTPException(404)
    return J

@app.delete("/api/video/job/{jid}")
def video_cancel(jid: str):
    if jid in VJOBS: VJOBS[jid]["cancel"] = True
    return {"ok": True}

@app.post("/api/video/result/add")
def video_result_add(request: Request, d: dict):
    """Analiz sonucundaki yuzu bir kisinin galerisine ekle."""
    need(request, "mudur_yrd")
    J = VJOBS.get(d.get("job", "")); i = d.get("i", -1)
    if not J or not (0 <= i < len(J["results"])): raise HTTPException(404)
    r = J["results"][i]; name = safe_name(d.get("name") or r["name"] or "")
    if not (PEOPLE / name).exists(): raise HTTPException(400, "Kişi bulunamadı")
    img = cv2.imdecode(np.frombuffer(base64.b64decode(r["img"].split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    faces = E.detect(img)
    if not len(faces): raise HTTPException(400, "Bu karede yüz net değil")
    _save_face(name, img, max(faces, key=lambda x: x[2] * x[3])); train_bg()
    audit(request, "videodan galeriye eklendi", name); return {"ok": True}

# ---- ziyaretciler
VISIT_ALERTS = []

def visitors_all():
    out = []
    for d in sorted(PEOPLE.iterdir()):
        if d.is_dir() and liste_of(d.name) == "ziyaretci":
            i = read_info(d.name)
            out.append({"name": d.name, **i, "photos": sorted(p.name for p in d.glob("*.jpg")),
                        "gecikti": i.get("durum") == "iceride" and i.get("bitis", "9") < datetime.now().strftime("%Y-%m-%d %H:%M")})
    return out

def save_info(name, info):
    (PEOPLE / name / "info.json").write_text(json.dumps(info, ensure_ascii=False), "utf-8"); INFO_CACHE.pop(name, None)

@app.get("/api/visitors")
def visitors(request: Request):
    v = visitors_all(); v.sort(key=lambda x: (x.get("durum") != "iceride", x.get("kayit", "")), reverse=False)
    return {"visitors": v, "alerts": VISIT_ALERTS[:10]}

@app.post("/api/visitors")
async def visitor_add(request: Request, ad: str = Form(...), tel: str = Form(""), ziyaret: str = Form(""), neden: str = Form(""),
                      sure: float = Form(2), cam: str = Form(""), file: UploadFile | None = File(None)):
    """Ziyaretci kaydi: foto (yukleme ya da kameradan son kare) + sureli izin."""
    name = safe_name(ad)
    if (PEOPLE / name).exists() and liste_of(name) != "ziyaretci": raise HTTPException(400, "Bu isimde öğrenci/personel kaydı var, farklı yaz")
    img = None
    if file is not None and file.filename:
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    elif SNAP["f"] is not None: img = SNAP["f"]
    else:
        c = CAMS.get(cam) or next((x for x in CAMS.values() if x.frame is not None), None)
        if c is not None:
            with c.lock: img = None if c.frame is None else c.frame.copy()
    if img is None: raise HTTPException(400, "Fotoğraf yok: kameradan görüntü al ya da fotoğraf yükle")
    faces = E.detect(img)
    if not len(faces): raise HTTPException(400, "Fotoğrafta yüz bulunamadı")
    (PEOPLE / name).mkdir(exist_ok=True)
    _save_face(name, img, max(faces, key=lambda r: r[2] * r[3]))
    now = datetime.now()
    info = {"liste": "ziyaretci", "tel": tel.strip(), "ziyaret": ziyaret.strip(), "neden": neden.strip(), "durum": "iceride",
            "kayit": now.strftime("%Y-%m-%d %H:%M"), "bitis": (now + timedelta(hours=max(0.25, min(sure, 24)))).strftime("%Y-%m-%d %H:%M"),
            "kaydeden": request.state.u["user"]}
    save_info(name, info); SNAP["f"] = None; G.train()
    log_event(name, 1.0, "kayıt", "giris", kaynak="ziyaretçi kaydı", force=True)
    audit(request, "ziyaretçi kaydı", name, f"{info['ziyaret']} · {sure} saat")
    return {"ok": True, "bitis": info["bitis"]}

@app.post("/api/visitors/{name}/exit")
def visitor_exit(request: Request, name: str):
    name = safe_name(name); i = read_info(name)
    if i.get("liste") != "ziyaretci": raise HTTPException(404)
    i["durum"] = "cikti"; i["cikis"] = datetime.now().strftime("%Y-%m-%d %H:%M"); save_info(name, i)
    log_event(name, 1.0, "elle:" + request.state.u["user"], "cikis", kaynak="elle", force=True)
    audit(request, "ziyaretçi çıkışı", name); return {"ok": True}

@app.post("/api/visitors/{name}/extend")
def visitor_extend(request: Request, name: str, d: dict):
    name = safe_name(name); i = read_info(name)
    if i.get("liste") != "ziyaretci": raise HTTPException(404)
    base = max(datetime.now(), datetime.strptime(i["bitis"], "%Y-%m-%d %H:%M"))
    i["bitis"] = (base + timedelta(hours=float(d.get("saat", 1)))).strftime("%Y-%m-%d %H:%M"); i.pop("uyarildi", None); save_info(name, i)
    VISIT_ALERTS[:] = [a for a in VISIT_ALERTS if a["name"] != name]
    audit(request, "ziyaret süresi uzatıldı", name, i["bitis"]); return {"ok": True, "bitis": i["bitis"]}

def visitor_watch():
    """Suresi dolup hala iceride olan ziyaretciyi uyar; cikmis ziyaretcinin verisini 1 gun sonra sil (KVKK)."""
    while True:
        try:
            now = datetime.now(); ns = now.strftime("%Y-%m-%d %H:%M"); changed = False
            for v in visitors_all():
                n = v["name"]
                if v.get("durum") == "iceride" and v.get("bitis", "9") < ns and not v.get("uyarildi"):
                    i = read_info(n); i["uyarildi"] = ns; save_info(n, i)
                    VISIT_ALERTS.insert(0, {"name": n, "bitis": v["bitis"], "ziyaret": v.get("ziyaret", ""), "time": ns}); del VISIT_ALERTS[30:]
                    wa_send(S["alarm_tel"], n, "ziyaretci", f"⚠ Ziyaretçi süresi doldu: {n} ({v.get('ziyaret', '')}) {v['bitis'][11:]} itibarıyla hâlâ içeride görünüyor.", force_text=True)
                    q("INSERT INTO audit(time,user,action,target,detail) VALUES(?,?,?,?,?)", (now_s(), "sistem", "ziyaretçi süresi doldu", n, v["bitis"]))
                ref = v.get("cikis") or v.get("bitis") or v.get("kayit")
                if ref and datetime.strptime(ref, "%Y-%m-%d %H:%M") < now - timedelta(days=1) and (v.get("durum") == "cikti" or ref < (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M")):
                    shutil.rmtree(PEOPLE / n, ignore_errors=True); INFO_CACHE.pop(n, None); changed = True
                    q("INSERT INTO audit(time,user,action,target,detail) VALUES(?,?,?,?,?)", (now_s(), "sistem", "ziyaretçi verisi silindi (KVKK)", n, ""))
            if changed: G.train()
        except Exception as e: print("ziyaretci hatasi:", e)
        time.sleep(60)
threading.Thread(target=visitor_watch, daemon=True).start()

# ================= haftalik rapor (PDF) =================
class _AdminReq:  # raporlari arka planda (istek olmadan) uretmek icin
    class state: u = {"user": "sistem", "rol": "yonetici", "sinif": "", "ad": "sistem"}

def _pdf_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    if "HV" in pdfmetrics.getRegisteredFontNames(): return "HV", "HVB"
    cands = [(ROOT / "fonts" / "DejaVuSans.ttf", ROOT / "fonts" / "DejaVuSans-Bold.ttf"),
             (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
             (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
             (Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"), Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"))]
    for r, b in cands:
        if r.exists():
            pdfmetrics.registerFont(TTFont("HV", str(r))); pdfmetrics.registerFont(TTFont("HVB", str(b if b.exists() else r)))
            return "HV", "HVB"
    return "Helvetica", "Helvetica-Bold"

def build_report_pdf(start, end, sinif=""):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import ParagraphStyle
    R = reports(_AdminReq(), start=start, end=end, sinif=sinif)
    F, FB = _pdf_font()
    H = ParagraphStyle("h", fontName=FB, fontSize=18, leading=22, textColor=colors.HexColor("#0b2d55"))
    H2 = ParagraphStyle("h2", fontName=FB, fontSize=12, leading=16, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#0f172a"))
    P = ParagraphStyle("p", fontName=F, fontSize=9.5, leading=13, textColor=colors.HexColor("#475569"))
    fd = lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%d.%m.%Y")
    st = R["students"]; n = len(st)
    avg = round(sum(s["oran"] for s in st) / n, 1) if n else 0
    late = sum(s["gec"] for s in st); absent = sum(s["gelmedi"] for s in st)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title="Devam Raporu", author="Hype Vision")
    el = []
    logo = ROOT / "logo" / "darklogo.png"
    if logo.exists(): el += [RLImage(str(logo), width=42 * mm, height=7.4 * mm, hAlign="LEFT"), Spacer(1, 6)]
    el += [Paragraph(f"{S.get('okul_adi') or 'Okul'} — Devam Raporu", H),
           Paragraph(f"{fd(R['start'])} – {fd(R['end'])} · {R['school_days']} okul günü · {n} öğrenci{(' · ' + sinif) if sinif else ''}", P), Spacer(1, 10)]
    def tbl(rows, widths, head=True, zebra=True):
        t = Table(rows, colWidths=widths, repeatRows=1 if head else 0)
        sty = [("FONT", (0, 0), (-1, -1), F, 9), ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
               ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#e7ebf1"))]
        if head: sty += [("FONT", (0, 0), (-1, 0), FB, 9), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569"))]
        t.setStyle(TableStyle(sty)); return t
    kpi = Table([[f"%{avg}", str(late), str(absent), str(sum(1 for s in st if s['oran'] < 80))],
                 ["Ortalama devam", "Geç giriş", "Devamsız gün (toplam)", "Devamı %80 altı"]], colWidths=[44 * mm] * 4)
    kpi.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), FB, 18), ("FONT", (0, 1), (-1, 1), F, 8.5), ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#64748b")),
                             ("TEXTCOLOR", (0, 0), (0, 0), colors.HexColor("#16a34a")), ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor("#7c3aed")),
                             ("TEXTCOLOR", (2, 0), (2, 0), colors.HexColor("#d97706")), ("TEXTCOLOR", (3, 0), (3, 0), colors.HexColor("#dc2626")),
                             ("BOX", (0, 0), (0, 1), 0.6, colors.HexColor("#e7ebf1")), ("BOX", (1, 0), (1, 1), 0.6, colors.HexColor("#e7ebf1")),
                             ("BOX", (2, 0), (2, 1), 0.6, colors.HexColor("#e7ebf1")), ("BOX", (3, 0), (3, 1), 0.6, colors.HexColor("#e7ebf1")),
                             ("TOPPADDING", (0, 0), (-1, -1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    el += [kpi, Paragraph("Sınıflar", H2)]
    el.append(tbl([["Sınıf", "Öğrenci", "Ortalama devam", "Geç giriş"]] + [[c["sinif"], c["ogrenci"], f"%{c['oran']}", c["gec"]] for c in R["classes"]],
                  [40 * mm, 35 * mm, 50 * mm, 40 * mm]))
    el.append(Paragraph("Günlük gelen öğrenci", H2))
    el.append(tbl([["Gün", "Gelen", "Toplam", "Oran"]] + [[fd(d["day"]), d["gelen"], d["toplam"], f"%{round(100 * d['gelen'] / d['toplam'], 1) if d['toplam'] else 0}"] for d in R["daily"]],
                  [45 * mm, 35 * mm, 35 * mm, 35 * mm]))
    lows = sorted([s for s in st if s["gelmedi"]], key=lambda s: (-s["gelmedi"], s["name"]))[:25]
    el.append(Paragraph("En çok devamsızlık yapanlar", H2))
    el.append(tbl([["No", "Ad Soyad", "Sınıf", "Gelmedi (gün)", "Devam"]] + [[s["no"], s["name"], s["sinif"], s["gelmedi"], f"%{s['oran']}"] for s in lows] if lows else [["Bu dönemde devamsızlık yok"]],
                  [20 * mm, 70 * mm, 25 * mm, 30 * mm, 25 * mm] if lows else [170 * mm], head=bool(lows)))
    lates = sorted([s for s in st if s["gec"]], key=lambda s: (-s["gec"], s["name"]))[:25]
    el.append(Paragraph("En çok geç kalanlar", H2))
    el.append(tbl([["No", "Ad Soyad", "Sınıf", "Geç (gün)"]] + [[s["no"], s["name"], s["sinif"], s["gec"]] for s in lates] if lates else [["Bu dönemde geç giriş yok"]],
                  [20 * mm, 80 * mm, 30 * mm, 40 * mm] if lates else [170 * mm], head=bool(lates)))
    el += [Spacer(1, 12), Paragraph(f"Hype Vision tarafından {datetime.now():%d.%m.%Y %H:%M} tarihinde otomatik oluşturuldu.", P)]
    doc.build(el)
    return buf.getvalue(), R

@app.get("/api/reports/pdf")
def report_pdf(request: Request, start: str = "", end: str = "", sinif: str = ""):
    sinif = my_class(request) or sinif
    data, R = build_report_pdf(start or datetime.now().strftime("%Y-%m-01"), end or today_s(), sinif)
    fn = f"devam-raporu-{R['start']}_{R['end']}{('-' + sinif) if sinif else ''}.pdf"
    return Response(data, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{fn}"'})

def wa_send_doc(tel_raw, name, data, filename, caption):
    tel = wa_phone(tel_raw)
    if not tel or S["wa_provider"] != "whapi" or not S["whapi_token"]: return "desteklenmiyor"
    mid = ins("INSERT INTO messages(time,tel,name,kind,text,status) VALUES(?,?,?,?,?,?)", (now_s(), tel, name, "rapor", caption + f" [{filename}]", "gonderiliyor"))
    body = {"to": tel, "media": "data:application/pdf;name=" + urllib.parse.quote(filename) + ";base64," + base64.b64encode(data).decode(),
            "filename": filename, "caption": caption}
    req = urllib.request.Request("https://gate.whapi.cloud/messages/document", json.dumps(body).encode(),
                                 {"Authorization": "Bearer " + S["whapi_token"], "Content-Type": "application/json", "Accept": "application/json",
                                  "User-Agent": "HypeVision/1.0 (+okul-giris)"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=60).read())
        q("UPDATE messages SET status='gonderildi', wamid=? WHERE id=?", ((r.get("message") or {}).get("id", ""), mid)); return "gonderildi"
    except urllib.error.HTTPError as e:
        q("UPDATE messages SET status='hata', err=? WHERE id=?", (e.read().decode("utf-8", "ignore")[:300], mid)); return "hata"
    except Exception as e:
        q("UPDATE messages SET status='hata', err=? WHERE id=?", (str(e)[:300], mid)); return "hata"

def mail_send(to, subject, text, data=None, filename=None):
    import smtplib
    from email.message import EmailMessage
    if not (S.get("smtp_host") and to): return "kapali"
    m = EmailMessage(); m["Subject"] = subject; m["From"] = S.get("smtp_from") or S.get("smtp_user"); m["To"] = to; m.set_content(text)
    if data: m.add_attachment(data, maintype="application", subtype="pdf", filename=filename)
    port = int(S.get("smtp_port") or 587)
    try:
        with (smtplib.SMTP_SSL(S["smtp_host"], port, timeout=30) if port == 465 else smtplib.SMTP(S["smtp_host"], port, timeout=30)) as sv:
            if port != 465: sv.starttls()
            if S.get("smtp_user"): sv.login(S["smtp_user"], S.get("smtp_pass", ""))
            sv.send_message(m)
        return "gonderildi"
    except Exception as e: return "hata: " + str(e)[:120]

def send_weekly(start=None, end=None):
    if not start:
        t = datetime.now(); mon = t - timedelta(days=t.weekday() + 7)
        start, end = mon.strftime("%Y-%m-%d"), (mon + timedelta(days=6)).strftime("%Y-%m-%d")
    data, R = build_report_pdf(start, end)
    st = R["students"]; avg = round(sum(s["oran"] for s in st) / len(st), 1) if st else 0
    fn = f"haftalik-devam-{start}_{end}.pdf"
    cap = (f"📊 {S.get('okul_adi') or 'Okul'} haftalık devam raporu\n{datetime.strptime(start, '%Y-%m-%d'):%d.%m} – {datetime.strptime(end, '%Y-%m-%d'):%d.%m.%Y}\n"
           f"Ortalama devam: %{avg} · Geç giriş: {sum(s['gec'] for s in st)} · Devamsız gün: {sum(s['gelmedi'] for s in st)}")
    res = []
    for tel in [x.strip() for x in (S.get("rapor_tel") or "").split(",") if x.strip()]:
        st_ = wa_send_doc(tel, "Haftalık rapor", data, fn, cap)
        if st_ == "desteklenmiyor": st_ = wa_send(tel, "Haftalık rapor", "rapor", cap, force_text=True)
        res.append(f"WA {tel}: {st_}")
    for to in [x.strip() for x in (S.get("rapor_email") or "").split(",") if x.strip()]:
        res.append(f"E-posta {to}: {mail_send(to, 'Haftalık devam raporu ' + start + ' / ' + end, cap, data, fn)}")
    q("INSERT INTO audit(time,user,action,target,detail) VALUES(?,?,?,?,?)", (now_s(), "sistem", "haftalık rapor gönderildi", f"{start}/{end}", "; ".join(res)[:300]))
    return res

def weekly_scheduler():
    while True:
        try:
            t = datetime.now(); wk = t.strftime("%G-%V")
            if S.get("rapor_aktif") and t.weekday() == int(S.get("rapor_gun", 0)) and t.strftime("%H:%M") >= (S.get("rapor_saat") or "08:00") \
                    and S.get("rapor_son") != wk:
                S["rapor_son"] = wk; save_settings(); send_weekly()
        except Exception as e: print("rapor hatasi:", e)
        time.sleep(60)
threading.Thread(target=weekly_scheduler, daemon=True).start()

@app.post("/api/reports/weekly/send")
def weekly_send_now(request: Request, d: dict):
    need(request, "yonetici")
    if not (S.get("rapor_tel") or S.get("rapor_email")): raise HTTPException(400, "Önce rapor alıcısı (telefon ya da e-posta) gir")
    res = send_weekly(d.get("start") or None, d.get("end") or None); audit(request, "haftalık rapor elle gönderildi", "", "; ".join(res)[:200])
    return {"result": res}

# ================= e-Okul uyumlu disa aktarma =================
@app.get("/api/eokul")
def eokul_export(request: Request, start: str, end: str = "", sinif: str = "", gec_yarim: int = 1):
    """e-Okul 'Devamsızlık Girişi' ekranina kolay giris icin: sinif basina sayfa, okul no sirasi, gun gun D/Y isaretleri,
    ozet (ozursuz tam gun / yarim gun). Ayrica 'Gunluk Giris' sayfasi: o gun gelmeyenler sinif sinif."""
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    end = end or start
    sinif = my_class(request) or sinif
    R = attendance_range(request, start=start, end=end, sinif=sinif)
    rows = R["rows"]; days = sorted({r["day"] for r in rows})
    by_cls = {}
    for r in rows: by_cls.setdefault(r["sinif"] or "Sınıfsız", {}).setdefault((r["no"], r["name"]), {})[r["day"]] = r
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    hf = Font(bold=True, color="FFFFFF"); hfill = PatternFill("solid", fgColor="0B2D55"); thin = Side(style="thin", color="D0D7E2")
    bd = Border(left=thin, right=thin, top=thin, bottom=thin); ctr = Alignment(horizontal="center", vertical="center")
    redf = PatternFill("solid", fgColor="FEE2E2"); purf = PatternFill("solid", fgColor="EDE9FE")
    # 1) gunluk giris sayfasi (tek gun secildiyse en kullanisli olan)
    ws = wb.create_sheet("Günlük Giriş")
    ws.append(["Tarih", "Sınıf", "Okul No", "Adı Soyadı", "Devamsızlık Türü", "Süre", "Açıklama"])
    for day in days:
        for cls in sorted(by_cls):
            for (no, name), dd in sorted(by_cls[cls].items(), key=lambda x: (int(x[0][0]) if str(x[0][0]).isdigit() else 10**9, x[0][1])):
                r = dd.get(day)
                if not r: continue
                if r["durum"] == "gelmedi": ws.append([datetime.strptime(day, "%Y-%m-%d").strftime("%d.%m.%Y"), cls, no, name, "Özürsüz", "Tam Gün", ""])
                elif r["durum"] == "gec" and gec_yarim: ws.append([datetime.strptime(day, "%Y-%m-%d").strftime("%d.%m.%Y"), cls, no, name, "Özürsüz", "Yarım Gün", f"Geç giriş {r['giris']}"])
    # 2) sinif cizelgeleri
    for cls in sorted(by_cls):
        sh = wb.create_sheet(re.sub(r"[\\/*?:\[\]]", "-", cls)[:31])
        sh.append(["Okul No", "Adı Soyadı"] + [datetime.strptime(d, "%Y-%m-%d").strftime("%d.%m") for d in days] + ["Özürsüz Tam Gün", "Özürsüz Yarım Gün", "Toplam (gün)"])
        for (no, name), dd in sorted(by_cls[cls].items(), key=lambda x: (int(x[0][0]) if str(x[0][0]).isdigit() else 10**9, x[0][1])):
            line = [int(no) if str(no).isdigit() else no, name]; tam = yarim = 0
            for d in days:
                r = dd.get(d)
                if r and r["durum"] == "gelmedi": line.append("D"); tam += 1
                elif r and r["durum"] == "gec" and gec_yarim: line.append("Y"); yarim += 1
                else: line.append("")
            sh.append(line + [tam, yarim, tam + yarim * 0.5])
        for c in sh[1]: c.font = hf; c.fill = hfill; c.alignment = ctr; c.border = bd
        for row in sh.iter_rows(min_row=2):
            for c in row:
                c.border = bd
                if c.column > 2: c.alignment = ctr
                if c.value == "D": c.fill = redf
                if c.value == "Y": c.fill = purf
        sh.column_dimensions["A"].width = 9; sh.column_dimensions["B"].width = 26
        for i in range(len(days)): sh.column_dimensions[openpyxl.utils.get_column_letter(3 + i)].width = 6
        sh.freeze_panes = "C2"
    for c in ws[1]: c.font = hf; c.fill = hfill; c.alignment = ctr
    for col, w in zip("ABCDEFG", (12, 8, 9, 26, 16, 11, 20)): ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"
    info = wb.create_sheet("Açıklama")
    for line in ["Bu dosya e-Okul > Devamsızlık İşlemleri > Günlük Devamsızlık Girişi ekranına hızlı giriş için hazırlanmıştır.",
                 "D = Özürsüz tam gün devamsızlık (kamera ve yoklamada hiç görülmedi)",
                 "Y = Geç giriş (okulun yönetmeliğine göre 'yarım gün' ya da 'geç gelme' olarak girilebilir)",
                 "Özürlü (raporlu / izinli) devamsızlıklar okul tarafından e-Okul'da ayrıca işaretlenmelidir.",
                 "Öğrenciler e-Okul ekranındaki gibi okul numarası sırasındadır.",
                 f"Oluşturma: {datetime.now():%d.%m.%Y %H:%M} · Hype Vision"]:
        info.append([line])
    info.column_dimensions["A"].width = 120
    buf = io.BytesIO(); wb.save(buf)
    audit(request, "e-Okul dışa aktarma", sinif or "tüm okul", f"{start}/{end}")
    fn = f"eokul-devamsizlik-{start}{('_' + end) if end != start else ''}{('-' + sinif) if sinif else ''}.xlsx"
    return Response(buf.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{fn}"'})

# ================= coklu okul (merkez) =================
@app.get("/api/central/summary")
def central_summary(request: Request):
    """Merkez panelin cektigi ozet. X-Central-Token ile korunur; ogrenci adi/foto icermez (KVKK)."""
    tok = request.headers.get("x-central-token", "")
    if not S.get("central_token") or not secrets.compare_digest(tok, S["central_token"]): raise HTTPException(401, "Geçersiz merkez anahtarı")
    att = attendance(_AdminReq(), day="")
    tot = len(att); g = sum(1 for a in att if a["geldi"]); l = sum(1 for a in att if a["gec"])
    cams = [{"ad": c.name, "durum": "koptu" if c.down else c.status} for c in CAMS.values() if c.cfg.get("aktif", True)]
    cls = {}
    for a in att:
        c = cls.setdefault(a.get("sinif") or "—", [0, 0]); c[0] += 1; c[1] += bool(a["geldi"])
    week = reports(_AdminReq(), start=(datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d"), end=today_s())
    return {"okul": S.get("okul_adi") or "Okul", "time": now_s(), "off_day": is_off_day(),
            "toplam": tot, "geldi": g, "gec": l, "gelmedi": tot - g, "oran": round(100 * g / tot, 1) if tot else 0,
            "kameralar": cams, "kamera_sorun": sum(1 for c in cams if c["durum"] != "canli"),
            "onay_bekleyen": q("SELECT COUNT(*) n FROM pending WHERE kind='onay'", one=True)["n"],
            "kara_alarm_bugun": q("SELECT COUNT(*) n FROM events WHERE liste='kara' AND time LIKE ?", (today_s() + "%",), one=True)["n"],
            "ziyaretci_iceride": sum(1 for v in visitors_all() if v.get("durum") == "iceride"),
            "siniflar": [{"sinif": k, "toplam": v[0], "geldi": v[1]} for k, v in sorted(cls.items())],
            "hafta": [{"day": d["day"], "oran": round(100 * d["gelen"] / d["toplam"], 1) if d["toplam"] else 0} for d in week["daily"]],
            "surum": "hypevision-okul"}

@app.get("/api/central")
def central(request: Request):
    need(request, "yonetici")
    import concurrent.futures as cf
    def fetch(o):
        url = o["url"].rstrip("/") + "/api/central/summary"
        try:
            req = urllib.request.Request(url, headers={"X-Central-Token": o.get("token", ""), "User-Agent": "HypeVision-Merkez/1.0"})
            ctx = None
            if url.startswith("https") and o.get("insecure"):
                import ssl; ctx = ssl._create_unverified_context()
            r = json.loads(urllib.request.urlopen(req, timeout=8, context=ctx).read()); r["ok"] = True
        except urllib.error.HTTPError as e: r = {"ok": False, "err": f"HTTP {e.code}"}
        except Exception as e: r = {"ok": False, "err": str(e)[:120]}
        r["_id"] = o["id"]; r["_ad"] = o.get("ad") or r.get("okul") or o["url"]; r["_url"] = o["url"]
        return r
    okullar = S.get("okullar", [])
    with cf.ThreadPoolExecutor(max_workers=8) as ex: res = list(ex.map(fetch, okullar))
    ok = [r for r in res if r["ok"]]
    tot = sum(r["toplam"] for r in ok); g = sum(r["geldi"] for r in ok)
    return {"schools": res, "total": {"okul": len(okullar), "erisilen": len(ok), "toplam": tot, "geldi": g, "gec": sum(r["gec"] for r in ok),
                                      "oran": round(100 * g / tot, 1) if tot else 0, "kamera_sorun": sum(r["kamera_sorun"] for r in ok),
                                      "onay_bekleyen": sum(r["onay_bekleyen"] for r in ok), "kara": sum(r["kara_alarm_bugun"] for r in ok)}}

@app.post("/api/central/schools")
def central_save(request: Request, d: dict):
    need(request, "yonetici")
    url = (d.get("url") or "").strip().rstrip("/")
    if not url.startswith(("http://", "https://")): raise HTTPException(400, "Adres http:// ya da https:// ile başlamalı")
    lst = S.setdefault("okullar", []); sid = d.get("id") or secrets.token_hex(4)
    o = {"id": sid, "ad": (d.get("ad") or "").strip(), "url": url, "token": (d.get("token") or "").strip(), "insecure": bool(d.get("insecure"))}
    i = next((k for k, x in enumerate(lst) if x["id"] == sid), None)
    if i is None: lst.append(o)
    else:
        if not o["token"]: o["token"] = lst[i].get("token", "")
        lst[i] = o
    save_settings(); audit(request, "merkeze okul eklendi", o["ad"] or url); return {"ok": True}

@app.delete("/api/central/schools/{sid}")
def central_del(request: Request, sid: str):
    need(request, "yonetici"); S["okullar"] = [x for x in S.get("okullar", []) if x["id"] != sid]; save_settings(); return {"ok": True}

@app.post("/api/central/token")
def central_token(request: Request):
    """Bu okulun merkeze baglanmasi icin anahtar uret."""
    need(request, "yonetici"); S["central_token"] = secrets.token_urlsafe(24); save_settings()
    audit(request, "merkez anahtarı üretildi"); return {"token": S["central_token"]}

# ================= kamera sihirbazi: RTSP olusturma, test, ag taramasi =================
import socket as _sock
RTSP_TPL = {
    "dahua":     ["/cam/realmonitor?channel={ch}&subtype={st}"],
    "hikvision": ["/Streaming/Channels/{ch}0{st1}", "/h264/ch{ch}/{ms}/av_stream"],
    "uniview":   ["/unicast/c{ch}/s{st}/live", "/media/video{st1}"],
    "axis":      ["/axis-media/media.amp", "/axis-media/media.amp?camera={ch}"],
    "tplink":    ["/stream{st1}"],
    "reolink":   ["/h264Preview_0{ch}_{ms}", "/Preview_0{ch}_{ms}"],
    "ipwebcam":  ["/h264.sdp", "/h264_ulaw.sdp", "/h264_aac.sdp", "/h264_opus.sdp"],
    "genel":     ["/live", "/stream1", "/11", "/live/ch{ch}", "/onvif1", "/videoMain", "/1", "/h264.sdp", "/video"],
}
BRAND_ORDER = ["dahua", "hikvision", "uniview", "axis", "tplink", "reolink", "ipwebcam", "genel"]

def build_rtsp(marka, ip, user, pw, ch=1, sub=0, port=554, idx=0):
    marka = marka if marka in RTSP_TPL else "genel"
    path = RTSP_TPL[marka][idx].format(ch=int(ch), st=int(sub), st1=int(sub) + 1, ms="sub" if sub else "main")
    cred = f"{urllib.parse.quote(user, safe='')}:{urllib.parse.quote(pw, safe='')}@" if user else ""
    return f"rtsp://{cred}{ip.strip()}:{int(port)}{path}"

def tcp_open(ip, port, t=1.5):
    try:
        with _sock.create_connection((ip, int(port)), timeout=t): return True
    except OSError: return False

def try_open(url, timeout=7):
    """RTSP'yi zaman asimli ac, bir kare oku. Donus: (ok, frame|None)."""
    res = {}
    def run():
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000])
            ok = cap.isOpened(); f = None
            if ok:
                for _ in range(3):
                    ok, f = cap.read()
                    if ok: break
            res["v"] = (bool(ok and f is not None), f, cap.get(cv2.CAP_PROP_FPS) if ok else 0); cap.release()
        except Exception: res["v"] = (False, None, 0)
    th = threading.Thread(target=run, daemon=True); th.start(); th.join(timeout + 3)
    return res.get("v", (False, None, 0))

def _cam_result(url, f, fps):
    sm = f if f.shape[1] <= 960 else cv2.resize(f, None, fx=960 / f.shape[1], fy=960 / f.shape[1])
    faces = E.detect(sm)
    return {"ok": True, "url": url, "safe_url": re.sub(r"//[^@/]*@", "//***@", url), "w": int(f.shape[1]), "h": int(f.shape[0]),
            "fps": round(float(fps or 0), 1), "faces": len(faces), "img": b64jpg(sm, 80)}

@app.post("/api/cameras/test")
def camera_test(request: Request, d: dict):
    """Tek adresi dene; ya da marka/IP/kullanici/sifre/kanal ile otomatik dene (bilinen tum yollar)."""
    need(request, "mudur_yrd")
    url = (d.get("url") or "").strip()
    if url.isdigit():
        cap = open_webcam(int(url)); ok, f = cap.read() if cap.isOpened() else (False, None); cap.release()
        if ok: return _cam_result(url, f, 0)
        return {"ok": False, "err": f"{url} numaralı webcam açılamadı (başka bir program kullanıyor olabilir)"}
    ip = (d.get("ip") or "").strip(); port = int(d.get("port") or 554)
    if url.startswith("http") and ("onvif" in url or url.rstrip("/").count("/") <= 2):
        # ONVIF / web arayuzu adresi yapistirildi: ayni host:port uzerinde bilinen RTSP yollarini dene
        m = re.match(r"https?://(?:[^@/]*@)?([^:/]+)(?::(\d+))?", url)
        if m:
            h, pp = m.group(1), int(m.group(2) or 80)
            for path in RTSP_TPL["ipwebcam"] + ["/live", "/stream1", "/h264", "/"]:
                ok, f, fps = try_open(f"rtsp://{h}:{pp}{path}", 4)
                if ok: r = _cam_result(f"rtsp://{h}:{pp}{path}", f, fps); r["not"] = "ONVIF adresi yerine RTSP görüntü adresi bulundu"; return r
            for path in ("/video", "/mjpg/video.mjpg", "/videostream.cgi"):
                ok, f, fps = try_open(f"http://{h}:{pp}{path}", 4)
                if ok: return _cam_result(f"http://{h}:{pp}{path}", f, fps)
            return {"ok": False, "err": "Bu bir ONVIF/web adresi; görüntü adresi (RTSP) bulunamadı. Kameranın RTSP adresini girin."}
    if url and not ip:
        m = re.match(r"rtsp://(?:[^@/]*@)?([^:/]+)(?::(\d+))?", url)
        if m: ip, port = m.group(1), int(m.group(2) or 554)
    if ip:
        if not tcp_open(ip, port, 2):
            if tcp_open(ip, 80, 1.5) or tcp_open(ip, 443, 1.5):
                return {"ok": False, "err": f"{ip} cihazı ağda var ama RTSP portu ({port}) kapalı. Kamera arayüzünden RTSP'yi açın ya da portu kontrol edin."}
            return {"ok": False, "err": f"{ip} adresine ulaşılamıyor. IP yanlış, kamera kapalı ya da sunucu ile aynı ağda değil."}
    if url:
        ok, f, fps = try_open(url)
        if ok: return _cam_result(url, f, fps)
        if not d.get("auto"): return {"ok": False, "err": "RTSP portu açık ama görüntü alınamadı: kullanıcı adı / şifre, kanal ya da yol yanlış olabilir. 'Otomatik bul' ile dene."}
    if not ip: return {"ok": False, "err": "IP adresi gerekli"}
    marka = d.get("marka") or "genel"; ch = int(d.get("kanal") or 1); sub = int(d.get("akis") or 0)
    order = [marka] + [b for b in BRAND_ORDER if b != marka] if d.get("auto") else [marka]
    tried = []
    for b in order:
        for i in range(len(RTSP_TPL[b])):
            u = build_rtsp(b, ip, d.get("user", ""), d.get("pass", ""), ch, sub, port, i)
            if u in tried: continue
            tried.append(u)
            ok, f, fps = try_open(u, 5)
            if ok: r = _cam_result(u, f, fps); r["marka"] = b; r["denenen"] = len(tried); return r
    return {"ok": False, "err": f"{len(tried)} farklı RTSP yolu denendi, görüntü alınamadı. Kullanıcı adı/şifreyi ve kanal numarasını kontrol et.",
            "denenen": [re.sub(r"//[^@/]*@", "//***@", u) for u in tried]}

@app.post("/api/cameras/build")
def camera_build(request: Request, d: dict):
    need(request, "mudur_yrd")
    return {"url": build_rtsp(d.get("marka", "dahua"), d.get("ip", ""), d.get("user", ""), d.get("pass", ""),
                              d.get("kanal") or 1, d.get("akis") or 0, d.get("port") or 554)}

def _local_nets():
    ips = set()
    try:
        for inf in _sock.getaddrinfo(_sock.gethostname(), None, _sock.AF_INET): ips.add(inf[4][0])
    except Exception: pass
    try:
        s_ = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM); s_.connect(("8.8.8.8", 80)); ips.add(s_.getsockname()[0]); s_.close()
    except Exception: pass
    return sorted({".".join(i.split(".")[:3]) for i in ips if not i.startswith(("127.", "169.254."))})

def onvif_probe(wait=3.0):
    """ONVIF WS-Discovery: agdaki IP kameralar kendini bildirir (marka/model/IP)."""
    msg = ('<?xml version="1.0" encoding="UTF-8"?><e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
           'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
           'xmlns:dn="http://www.onvif.org/ver10/network/wsdl"><e:Header><w:MessageID>uuid:' + secrets.token_hex(16) +
           '</w:MessageID><w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
           '<w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>'
           '<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body></e:Envelope>').encode()
    found = {}
    try:
        s_ = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM, _sock.IPPROTO_UDP)
        s_.setsockopt(_sock.IPPROTO_IP, _sock.IP_MULTICAST_TTL, 2); s_.settimeout(0.5)
        for _ in range(2): s_.sendto(msg, ("239.255.255.250", 3702))
        end = time.time() + wait
        while time.time() < end:
            try: data, addr = s_.recvfrom(65535)
            except _sock.timeout: continue
            t = data.decode("utf-8", "ignore")
            scopes = " ".join(re.findall(r"<[^>]*Scopes[^>]*>([^<]*)<", t))
            name = re.search(r"onvif://www\.onvif\.org/name/([^\s]+)", scopes); hw = re.search(r"onvif://www\.onvif\.org/hardware/([^\s]+)", scopes)
            mk = (urllib.parse.unquote(name.group(1)) if name else "")
            model = urllib.parse.unquote(hw.group(1)) if hw else ""
            low = (mk + " " + model + " " + scopes).lower()
            marka = next((b for b in ("dahua", "hikvision", "uniview", "axis", "tplink", "reolink") if b in low or (b == "hikvision" and "hik" in low) or (b == "uniview" and "unv" in low)), "")
            found[addr[0]] = {"ip": addr[0], "ad": mk, "model": model, "marka": marka, "kaynak": "ONVIF"}
        s_.close()
    except Exception as e: print("onvif hata:", e)
    return found

@app.get("/api/cameras/discover")
def camera_discover(request: Request):
    """Agdaki kameralari bul: once ONVIF, sonra yerel /24 aginda 554 (RTSP) portu acik cihazlar."""
    need(request, "mudur_yrd")
    import concurrent.futures as cf
    found = onvif_probe(3.0)
    nets = _local_nets(); cand = [f"{n}.{i}" for n in nets[:3] for i in range(1, 255)]
    with cf.ThreadPoolExecutor(max_workers=128) as ex:
        for ip, ok in zip(cand, ex.map(lambda ip: tcp_open(ip, 554, 0.4), cand)):
            if ok and ip not in found: found[ip] = {"ip": ip, "ad": "", "model": "", "marka": "", "kaynak": "RTSP portu açık"}
            elif ok: found[ip]["rtsp"] = True
    used = {re.sub(r"rtsp://(?:[^@/]*@)?([^:/]+).*", r"\1", c.get("url", "")) for c in cams_cfg()}
    out = sorted(found.values(), key=lambda x: tuple(int(p) for p in x["ip"].split(".")))
    for o in out: o["ekli"] = o["ip"] in used
    return {"nets": nets, "cameras": out}

# ================= sifirlama =================
BACKUPS = DATA / "backups"
RESET_PARTS = {
    "model":     "Öğrenciler, fotoğraflar ve tanıma modeli (galeri)",
    "gecis":     "Geçiş / yoklama kayıtları ve geçiş anı görüntüleri",
    "bekleyen":  "Onay bekleyen ve bilinmeyen yüzler",
    "mesaj":     "WhatsApp mesaj geçmişi",
    "ziyaretci": "Ziyaretçiler",
    "islem":     "İşlem kaydı (audit)",
}

def make_backup_zip(path):
    with DBL: CONN.commit()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in PEOPLE.rglob("*"):
            if p.is_file(): z.write(p, "people/" + p.relative_to(PEOPLE).as_posix())
        for p in (GALLERY, SETTINGS, DB_PATH):
            if p.exists(): z.write(p, p.name)

@app.get("/api/reset/info")
def reset_info(request: Request):
    need(request, "yonetici")
    ppl = [d for d in PEOPLE.iterdir() if d.is_dir()]
    return {"parts": RESET_PARTS, "counts": {
        "model": sum(1 for d in ppl if liste_of(d.name) != "ziyaretci"),
        "foto": sum(len(list(d.glob("*.jpg"))) for d in ppl if liste_of(d.name) != "ziyaretci"),
        "gecis": q("SELECT COUNT(*) n FROM events", one=True)["n"],
        "bekleyen": q("SELECT COUNT(*) n FROM pending", one=True)["n"],
        "mesaj": q("SELECT COUNT(*) n FROM messages", one=True)["n"],
        "ziyaretci": sum(1 for d in ppl if liste_of(d.name) == "ziyaretci"),
        "islem": q("SELECT COUNT(*) n FROM audit", one=True)["n"]},
        "backups": sorted((p.name for p in BACKUPS.glob("*.zip")), reverse=True)[:10] if BACKUPS.exists() else []}

@app.post("/api/reset")
def reset(request: Request, d: dict):
    """Secilen verileri sil. Once otomatik yedek alinir. Kullanicilar, ayarlar ve kameralar korunur."""
    need(request, "yonetici")
    if (d.get("confirm") or "").strip().upper() != "SIFIRLA": raise HTTPException(400, 'Onay için "SIFIRLA" yaz')
    parts = [p for p in d.get("parts", []) if p in RESET_PARTS]
    if not parts: raise HTTPException(400, "Sıfırlanacak bir şey seçilmedi")
    BACKUPS.mkdir(exist_ok=True)
    bname = f"sifirlama-oncesi-{datetime.now():%Y%m%d-%H%M%S}.zip"; make_backup_zip(BACKUPS / bname)
    done = []
    for cam in CAMS.values():
        for tr in cam.tracks: tr.done = True  # yarim kalan izler eski isimle kayit dusmesin
        cam.tracks = []
    LAST_SEEN.clear(); ALARMS.clear(); VISIT_ALERTS.clear()
    if "model" in parts or "ziyaretci" in parts:
        for p in list(PEOPLE.iterdir()):
            if not p.is_dir(): continue
            is_v = read_info(p.name).get("liste") == "ziyaretci"
            if ("ziyaretci" in parts and is_v) or ("model" in parts and not is_v): shutil.rmtree(p, ignore_errors=True)
        INFO_CACHE.clear(); G.train()
        done += [RESET_PARTS[k] for k in ("model", "ziyaretci") if k in parts]
    if "gecis" in parts:
        q("DELETE FROM events"); shutil.rmtree(SNAPS, ignore_errors=True); SNAPS.mkdir(exist_ok=True); done.append(RESET_PARTS["gecis"])
    if "bekleyen" in parts:
        q("DELETE FROM pending")
        for f in (PENDING, UNKNOWN): shutil.rmtree(f, ignore_errors=True); f.mkdir(exist_ok=True)
        done.append(RESET_PARTS["bekleyen"])
    if "mesaj" in parts: q("DELETE FROM messages"); done.append(RESET_PARTS["mesaj"])
    if "islem" in parts: q("DELETE FROM audit"); done.append(RESET_PARTS["islem"])
    for f in UPLOADS.glob("*"): f.unlink(missing_ok=True)
    VENROLL.clear()
    with DBL: CONN.execute("VACUUM")
    audit(request, "SIFIRLAMA", ", ".join(parts), f"yedek: {bname}")
    return {"ok": True, "done": done, "backup": bname}

@app.get("/api/reset/backup/{name}")
def reset_backup_dl(request: Request, name: str):
    need(request, "yonetici")
    p = BACKUPS / safe_name(name)
    if not p.exists(): raise HTTPException(404)
    return Response(p.read_bytes(), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{p.name}"'})

@app.post("/api/people/bulk-delete")
def people_bulk_delete(request: Request, d: dict):
    """Toplu silme (ornegin ayrilan / mezun 10 ogrenci). Silinenler bir daha taninmaz; model kendiliginden guncellenir."""
    need(request, "mudur_yrd")
    names = [safe_name(n) for n in d.get("names", []) if n]
    if not names: raise HTTPException(400, "Kimse seçilmedi")
    purge = bool(d.get("purge")); n_ok = 0
    for name in names:
        if not (PEOPLE / name).exists(): continue
        shutil.rmtree(PEOPLE / name, ignore_errors=True); INFO_CACHE.pop(name, None); n_ok += 1
        if purge:
            for r in q("SELECT id FROM events WHERE name=? AND snap IS NOT NULL", (name,)):
                for suf in ("", "_f"): (SNAPS / f"{r['id']}{suf}.jpg").unlink(missing_ok=True)
            q("DELETE FROM events WHERE name=?", (name,)); q("DELETE FROM messages WHERE name=?", (name,))
            for r in q("SELECT id FROM pending WHERE cand=?", (name,)): drop_pending(r["id"])
    for cam in CAMS.values(): cam.tracks = []
    G.train()
    audit(request, "toplu kalıcı silme" if purge else "toplu silme", f"{n_ok} kişi", ", ".join(names)[:300])
    return {"ok": True, "deleted": n_ok}

# ---- test
@app.post("/api/test")
async def test_image(file: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "okunamadı")
    faces = E.detect(img)
    res = []
    crops = [norm_crop(img, f[4:14].reshape(5, 2)) for f in faces]
    for fc, crop, m in zip(faces, crops, G.match_full(E.embed(crops))):
        qv = face_quality(crop, fc); acc, why = decide(m, qv); n, s = m["name"], m["score"]
        ok = acc is not None; x, y, w, h = fc[:4].astype(int)
        col = (0, 200, 0) if ok else (0, 165, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 3)
        cv2.putText(img, f"{ascii_name(n) if ok else '?'} {s:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        res.append({"name": n if ok else None, "best": n, "score": round(s, 3), "liste": liste_of(n) if ok else None,
                    "second": m["name2"], "score2": round(m["score2"], 3), "quality": qv, "why": why})
    return {"faces": res, "image": "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()}

apply_cameras()
if len(G.P) == 0 and any(PEOPLE.iterdir()): threading.Thread(target=G.train, daemon=True).start()
if q("SELECT 1 FROM pending WHERE kind='bilinmeyen' AND (emb IS NULL OR grp IS NULL) LIMIT 1"):
    threading.Thread(target=recluster, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
