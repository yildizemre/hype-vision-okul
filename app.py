"""Hype Vision - Okul Giris (yuz tanima) sunucusu.
Calistir:  venv\\Scripts\\python app.py   ->  http://localhost:8000
Modeller:  models/yunet.onnx, models/auraface/glintr100.onnx
Veri:      data/  (people/, gallery.npz, okul.db, settings.json, pending/, unknown/)
"""
import os, time, threading, json, re, shutil, sqlite3, hashlib, secrets, io, zipfile, base64, csv
import urllib.request, urllib.parse
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
PEOPLE, PENDING, UNKNOWN = DATA / "people", DATA / "pending", DATA / "unknown"
GALLERY, SETTINGS, DB_PATH = DATA / "gallery.npz", DATA / "settings.json", DATA / "okul.db"
for d in (PEOPLE, PENDING, UNKNOWN): d.mkdir(parents=True, exist_ok=True)

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
    "keep_days": 90,        # gecis/mesaj kayitlari saklama
    "keep_unknown_days": 1, # bilinmeyen/onay fotolari saklama
    "wa_enabled": False, "wa_token": "", "wa_phone_id": "", "wa_verify": "hypevision",
    "wa_mode": "text",      # text | template
    "wa_template": "Sayın {veli}, öğrenciniz {ogrenci} okula giriş yaptı. Saat: {saat}",
    "wa_tpl_name": "", "wa_tpl_lang": "tr",
    "alarm_tel": "",
    "wa_provider": "whapi",  # whapi | meta
    "whapi_token": "",
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
""")
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
    def __init__(self):
        self.names, self.embs = [], np.zeros((0, 512), np.float32)
        if GALLERY.exists():
            z = np.load(GALLERY, allow_pickle=False); self.names, self.embs = list(z["names"]), z["embs"]
        self.lock = threading.Lock()

    def train(self):
        with self.lock:
            names, embs, report = [], [], []
            for d in sorted(PEOPLE.iterdir()):
                if not d.is_dir(): continue
                crops = [c for c in (imread_u(p) for p in sorted(d.glob("*.jpg"))) if c is not None]
                if not crops: continue
                e = E.embed(crops); m = e.mean(0); m /= np.linalg.norm(m)
                names.append(d.name); embs.append(m)
                report.append({"name": d.name, "photos": len(crops), "min_sim": round(float((e @ m).min()), 3)})
            self.names = names
            self.embs = np.array(embs, np.float32) if embs else np.zeros((0, 512), np.float32)
            np.savez(GALLERY, names=np.array(names), embs=self.embs)
            risk = None
            if len(names) > 1:
                sim = self.embs @ self.embs.T; np.fill_diagonal(sim, -1)
                i, j = np.unravel_index(sim.argmax(), sim.shape)
                risk = {"a": names[i], "b": names[j], "sim": round(float(sim[i, j]), 3)}
            return {"people": report, "closest_pair": risk}

    def match(self, embs):
        if len(self.names) == 0 or len(embs) == 0: return [(None, 0.0)] * len(embs)
        sim = embs @ self.embs.T; idx = sim.argmax(1)
        return [(self.names[i], float(sim[k, i])) for k, i in enumerate(idx)]
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
                                     {"Authorization": "Bearer " + tok, "Content-Type": "application/json", "Accept": "application/json"})
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

# ================= olay kaydi =================
LAST_SEEN = {}
ALARMS = []  # canli alarm akisi (son 50)

def log_event(name, score, cam, kaynak="kamera", force=False):
    now = time.time()
    if not force and now - LAST_SEEN.get(name, 0) < S["cooldown"]: return None
    LAST_SEEN[name] = now
    info, t = read_info(name), now_s()
    lst = liste_of(name)
    eid = ins("INSERT INTO events(time,name,score,cam,sinif,liste,wa,kaynak) VALUES(?,?,?,?,?,?,?,?)",
              (t, name, round(score, 3), cam, info.get("sinif", ""), lst, "", kaynak))
    if lst == "kara":
        ALARMS.insert(0, {"id": eid, "time": t, "name": name, "not": info.get("not", ""), "cam": cam}); del ALARMS[50:]
        wa = wa_send(S["alarm_tel"], name, "alarm", f"⚠ KARA LİSTE: {name} kapıda görüldü. Saat {t[11:16]}. {info.get('not', '')}", eid, )
    elif lst == "beyaz": wa = "-"
    else: wa = wa_send(info.get("veli_tel"), name, "geldi", wa_vars(info, name, t), eid)
    q("UPDATE events SET wa=? WHERE id=? AND wa=''", (wa, eid))
    return eid

def add_pending(kind, cand, score, thumb, crop, cam):
    pid = ins("INSERT INTO pending(time,kind,cand,score,img,cam) VALUES(?,?,?,?,?,?)",
              (now_s(), kind, cand or "", round(score, 3), "", cam))
    folder = PENDING if kind == "onay" else UNKNOWN
    imwrite_u(folder / f"{pid}.jpg", thumb); imwrite_u(folder / f"{pid}_crop.jpg", crop)
    q("UPDATE pending SET img=? WHERE id=?", (f"{pid}.jpg", pid))

# ================= kamera + takip/oylama =================
def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    i = ix * iy; return i / (aw * ah + bw * bh - i + 1e-6)

class Track:
    _n = 0
    def __init__(self, box):
        Track._n += 1; self.id = Track._n; self.box = box; self.votes = []; self.last = time.time()
        self.done = False; self.decided = None; self.best = (0.0, None, None, None)  # score, name, thumb, crop
    def label(self):
        if self.decided: return self.decided
        if not self.votes: return None
        n, c = Counter(v[0] for v in self.votes).most_common(1)[0]
        return n

class Camera:
    def __init__(self):
        self.url = None; self.frame = None; self.annot = None; self.snap = None
        self.running = False; self.fps = 0.0; self.status = "kapali"; self.tracks = []
        self.lock = threading.Lock(); self.gen = 0

    def start(self, url):
        self.stop(); self.url = url; self.running = True; self.status = "baglaniyor"; self.gen += 1
        g = self.gen
        threading.Thread(target=self._reader, args=(g,), daemon=True).start()
        threading.Thread(target=self._worker, args=(g,), daemon=True).start()

    def stop(self):
        self.running = False; self.gen += 1; time.sleep(0.3)
        self.frame = None; self.annot = None; self.status = "kapali"; self.tracks = []

    def url_short(self): return re.sub(r"//[^@/]*@", "//", self.url or "")

    def _reader(self, g):
        src = int(self.url) if self.url.isdigit() else self.url
        while self.running and g == self.gen:
            cap = cv2.VideoCapture(src) if isinstance(src, int) else cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if not cap.isOpened(): self.status = "baglanamadi, tekrar deneniyor"; time.sleep(2); continue
            self.status = "canli"
            while self.running and g == self.gen:
                ok, f = cap.read()
                if not ok: self.status = "akis koptu"; break
                with self.lock: self.frame = f
            cap.release()

    def _finish(self, tr):
        """Takip bitti; karar verilmediyse onay/bilinmeyen kutusuna at."""
        if tr.done or len(tr.votes) < 2: return
        sc, name, thumb, crop = tr.best
        if thumb is None: return
        cam = self.url_short()
        if sc >= S["low_threshold"]: add_pending("onay", name, sc, thumb, crop, cam)
        else: add_pending("bilinmeyen", name, sc, thumb, crop, cam)

    def _worker(self, g):
        t0, n = time.time(), 0
        while self.running and g == self.gen:
            with self.lock: f = None if self.frame is None else self.frame.copy()
            if f is None: time.sleep(0.05); continue
            scale = 1.0; img = f
            if f.shape[1] > 1280: scale = 1280 / f.shape[1]; img = cv2.resize(f, None, fx=scale, fy=scale)
            faces = E.detect(img)
            dets, crops = [], []
            for fc in faces:
                fc = fc.copy(); fc[:14] /= scale
                if min(fc[2], fc[3]) < S["min_face"]: dets.append((fc, None)); continue
                crops.append(norm_crop(f, fc[4:14].reshape(5, 2))); dets.append((fc, len(crops) - 1))
            matches = G.match(E.embed(crops))
            now = time.time(); vis = f.copy(); used = set()
            for fc, ci in dets:
                box = tuple(fc[:4])
                if ci is None:
                    x, y, w, h = map(int, box); cv2.rectangle(vis, (x, y), (x + w, y + h), (150, 150, 150), 1); continue
                name, sc = matches[ci]
                tr = max((t for t in self.tracks if id(t) not in used), key=lambda t: iou(t.box, box), default=None)
                if tr is None or iou(tr.box, box) < 0.2: tr = Track(box); self.tracks.append(tr)
                used.add(id(tr)); tr.box = box; tr.last = now
                ok = name is not None and sc >= S["threshold"]
                tr.votes.append((name if ok else None, sc)); tr.votes = tr.votes[-15:]
                if sc > tr.best[0]: tr.best = (sc, name, face_thumb(f, fc), crops[ci])
                # oylama
                if not tr.done and len(tr.votes) >= S["vote_frames"]:
                    top, cnt = Counter(v[0] for v in tr.votes).most_common(1)[0]
                    if top and cnt >= 0.6 * len(tr.votes):
                        tr.done = True; tr.decided = top
                        mean = float(np.mean([v[1] for v in tr.votes if v[0] == top]))
                        log_event(top, mean, self.url_short())
                self._draw(vis, box, tr, name, sc)
            for tr in [t for t in self.tracks if now - t.last > 1.5]:
                self._finish(tr); self.tracks.remove(tr)
            self.annot = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            n += 1
            if now - t0 > 1: self.fps = n / (now - t0); t0, n = now, 0

    def _draw(self, vis, box, tr, name, sc):
        x, y, w, h = map(int, box); lab = tr.label()
        if lab:
            lst = liste_of(lab)
            col = (0, 0, 230) if lst == "kara" else (230, 140, 0) if lst == "beyaz" else (0, 190, 0)
            pre = "KARA LISTE: " if lst == "kara" else "[B] " if lst == "beyaz" else ""
            txt = f"{pre}{ascii_name(lab)} {sc:.2f}" + (" OK" if tr.done else f" {len(tr.votes)}/{int(S['vote_frames'])}")
        else:
            col = (0, 165, 255); txt = f"? {sc:.2f}"
        th = 4 if lab and liste_of(lab) == "kara" else 2
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, th)
        cv2.rectangle(vis, (x, y - 26), (x + max(w, 11 * len(txt)), y), col, -1)
        cv2.putText(vis, txt, (x + 4, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
CAM = Camera()

# ================= temizlik (saklama suresi) =================
def cleanup():
    while True:
        try:
            cut = (datetime.now() - timedelta(days=S["keep_days"])).strftime("%Y-%m-%d")
            q("DELETE FROM events WHERE time < ?", (cut,)); q("DELETE FROM messages WHERE time < ?", (cut,))
            ucut = (datetime.now() - timedelta(days=S["keep_unknown_days"])).strftime("%Y-%m-%d %H:%M:%S")
            for r in q("SELECT id,kind FROM pending WHERE time < ?", (ucut,)): drop_pending(r["id"], r["kind"])
            q("DELETE FROM tokens WHERE exp < ?", (time.time(),))
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
OPEN = {"/api/login", "/webhook/whatsapp"}

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
    return {"ok": True}

@app.delete("/api/users/{user}")
def del_user(request: Request, user: str):
    need(request, "yonetici")
    if user == request.state.u["user"]: raise HTTPException(400, "Kendini silemezsin")
    q("DELETE FROM users WHERE user=?", (user,)); q("DELETE FROM tokens WHERE user=?", (user,)); return {"ok": True}

@app.post("/api/password")
def change_pw(request: Request, d: dict):
    u = request.state.u
    if not check_pw(d.get("old", ""), u["pw"]): raise HTTPException(400, "Mevcut şifre yanlış")
    if len(d.get("new", "")) < 4: raise HTTPException(400, "Yeni şifre en az 4 karakter")
    q("UPDATE users SET pw=? WHERE user=?", (hpw(d["new"]), u["user"])); return {"ok": True}

# ================= sayfalar =================
@app.get("/", response_class=HTMLResponse)
def index(): return (ROOT / "index.html").read_text("utf-8")

@app.get("/logo/{f}")
def logo(f: str):
    p = ROOT / "logo" / safe_name(f)
    if not p.exists(): raise HTTPException(404)
    return Response(p.read_bytes(), media_type="image/png")

@app.get("/api/status")
def status(request: Request):
    pub = {k: v for k, v in S.items() if k not in ("wa_token", "whapi_token")} | {"wa_token_set": bool(S["wa_token"]), "whapi_token_set": bool(S["whapi_token"])}
    return {"camera": CAM.status, "url": CAM.url_short(), "fps": round(CAM.fps, 1), "provider": E.provider,
            "gallery": len(G.names), "settings": pub,
            "pending": len([r for r in q("SELECT cand FROM pending WHERE kind='onay'") if visible(request, r["cand"])]),
            "unknown": q("SELECT COUNT(*) n FROM pending WHERE kind='bilinmeyen'", one=True)["n"],
            "alarms": ALARMS[:5]}

@app.post("/api/settings")
def settings(request: Request, d: dict):
    need(request, "yonetici")
    for k in ("threshold", "low_threshold", "cooldown", "det_score", "min_face", "vote_frames", "keep_days", "keep_unknown_days"):
        if d.get(k) not in (None, ""): S[k] = float(d[k])
    for k in ("wa_phone_id", "wa_template", "wa_tpl_name", "wa_tpl_lang", "wa_mode", "alarm_tel", "wa_verify", "wa_provider"):
        if k in d: S[k] = str(d[k]).strip()
    if d.get("wa_token"): S["wa_token"] = str(d["wa_token"]).strip()
    if d.get("whapi_token"): S["whapi_token"] = str(d["whapi_token"]).strip()
    if "wa_enabled" in d: S["wa_enabled"] = bool(d["wa_enabled"])
    save_settings(); return {"ok": True}

# ---- kamera
@app.post("/api/camera")
def cam_start(request: Request, d: dict):
    need(request, "mudur_yrd")
    url = (d.get("url") or "").strip()
    if not url: raise HTTPException(400, "URL bos")
    CAM.start(url); return {"ok": True}

@app.delete("/api/camera")
def cam_stop(request: Request): need(request, "mudur_yrd"); CAM.stop(); return {"ok": True}

@app.get("/api/stream")
def stream():
    def gen():
        last = None
        while True:
            a = CAM.annot
            if a is not None and a is not last:
                last = a; yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + a + b"\r\n"
            time.sleep(0.03)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")

@app.get("/api/snapshot")
def snapshot():
    with CAM.lock: f = None if CAM.frame is None else CAM.frame.copy()
    if f is None: raise HTTPException(400, "Kamera görüntüsü yok")
    CAM.snap = f.copy()
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
    info["liste"] = d.get("liste") if d.get("liste") in ("ogrenci", "beyaz", "kara") else "ogrenci"
    (PEOPLE / name / "info.json").write_text(json.dumps(info, ensure_ascii=False), "utf-8")
    INFO_CACHE.pop(name, None)
    if old and old != name: train_bg()
    return {"ok": True}

@app.post("/api/enroll/capture")
def enroll_capture(request: Request, d: dict):
    need(request, "mudur_yrd")
    name = safe_name(d.get("name", "")); f = CAM.snap
    if f is None:
        with CAM.lock: f = None if CAM.frame is None else CAM.frame.copy()
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
    G.train(); return {"ok": True}

@app.post("/api/train")
def train(request: Request): need(request, "mudur_yrd"); return G.train()

# ---- gecisler / yoklama
@app.get("/api/events")
def events(request: Request, limit: int = 40, day: str = ""):
    rows = q("SELECT * FROM events WHERE time LIKE ? ORDER BY id DESC LIMIT ?", ((day or "") + "%", min(limit, 2000)))
    return [r for r in rows if visible(request, r["name"])]

@app.get("/api/attendance")
def attendance(request: Request, day: str = ""):
    day = day or datetime.now().strftime("%Y-%m-%d")
    seen = {}
    for r in q("SELECT name, MIN(time) t FROM events WHERE time LIKE ? GROUP BY name", (day + "%",)): seen[r["name"]] = r["t"][11:16]
    return [{"name": d.name, **read_info(d.name), "geldi": seen.get(d.name)}
            for d in sorted(PEOPLE.iterdir()) if d.is_dir() and liste_of(d.name) == "ogrenci" and visible(request, d.name)]

@app.post("/api/attendance/manual")
def manual(request: Request, d: dict):
    """Ogretmen elle 'geldi' isaretler (kamera kacirdiysa)."""
    name = safe_name(d.get("name", ""))
    if not visible(request, name): raise HTTPException(403, "Bu öğrenci senin sınıfında değil")
    log_event(name, 0, "elle:" + request.state.u["user"], kaynak="elle", force=True); return {"ok": True}

@app.post("/api/absent/notify")
def absent_notify(request: Request, d: dict):
    """Ogretmen teyidi sonrasi 'gelmedi' mesaji."""
    name = safe_name(d.get("name", ""))
    if not visible(request, name): raise HTTPException(403, "Bu öğrenci senin sınıfında değil")
    i = read_info(name)
    txt = f"Sayın {i.get('veli_ad') or 'Veli'}, öğrenciniz {name} bugün ({datetime.now():%d.%m.%Y}) okula gelmemiştir. Bilginize."
    return {"wa": wa_send(i.get("veli_tel"), name, "gelmedi", txt)}

# ---- onay bekleyen / bilinmeyen
@app.get("/api/pending")
def pending(request: Request, kind: str = "onay"):
    rows = q("SELECT * FROM pending WHERE kind=? ORDER BY id DESC LIMIT 300", (kind,))
    return [r for r in rows if kind != "onay" or visible(request, r["cand"])]

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
    drop_pending(pid); train_bg(); return {"ok": True}

@app.delete("/api/pending/{pid}")
def reject(request: Request, pid: int): drop_pending(pid); return {"ok": True}

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
    CAM.stop()
    shutil.rmtree(PEOPLE, ignore_errors=True); PEOPLE.mkdir()
    for n in names:
        if n.startswith("people/") and not n.endswith("/") and ".." not in n:
            t = PEOPLE / n[7:]; t.parent.mkdir(parents=True, exist_ok=True); t.write_bytes(z.read(n))
    if "settings.json" in names: SETTINGS.write_bytes(z.read("settings.json")); S.clear(); S.update(load_settings())
    if "okul.db" in names:
        with DBL:
            CONN.close(); DB_PATH.write_bytes(z.read("okul.db")); CONN = db()
    INFO_CACHE.clear(); G.train()
    return {"ok": True, "people": len(G.names)}

# ---- test
@app.post("/api/test")
async def test_image(file: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "okunamadı")
    faces = E.detect(img)
    res = []
    for fc, (n, s) in zip(faces, G.match(E.embed([norm_crop(img, f[4:14].reshape(5, 2)) for f in faces]))):
        ok = n is not None and s >= S["threshold"]; x, y, w, h = fc[:4].astype(int)
        col = (0, 200, 0) if ok else (0, 165, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 3)
        cv2.putText(img, f"{ascii_name(n) if ok else '?'} {s:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        res.append({"name": n if ok else None, "best": n, "score": round(s, 3), "liste": liste_of(n) if ok else None})
    return {"faces": res, "image": "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
