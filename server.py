#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foot Session — Serveur de tracking local (type LiveTag)
Détection YOLOv8 + tracker BoT-SORT (ré-identification) => suivi auto de TOUS les joueurs.
Tourne 100% en local. On analyse une PLAGE de temps choisie (une action), pas tout le match.
"""

import os, re, json, uuid, threading, traceback, subprocess, asyncio, shutil, time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).resolve().parent
# Dossier de TRAVAIL du moteur (cache des vidéos, rendus temporaires, sorties). Peut être placé sur un
# DISQUE EXTERNE via la variable MATCHLYZ_WORK_DIR -> le Mac ne se remplit plus avec les grosses vidéos.
# Repli AUTOMATIQUE sur le dossier local « jobs » si le chemin externe est absent/non inscriptible
# (disque débranché) -> l'app continue de fonctionner dans tous les cas.
def _pick_work_dir():
    ext = (os.environ.get("MATCHLYZ_WORK_DIR") or "").strip()
    # Fiabilité : si l'enveloppe (main.js) n'a pas passé la variable (ancienne app packagée), on lit
    # DIRECTEMENT le choix enregistré dans « engine-storage.txt » -> le disque externe est respecté
    # quel que soit le lanceur.
    if not ext:
        try:
            _sp = BASE / "engine-storage.txt"
            if _sp.exists():
                ext = _sp.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    if ext:
        try:
            p = Path(ext)
            p.mkdir(parents=True, exist_ok=True)
            _t = p / ".mlz_wtest"
            _t.write_text("ok"); _t.unlink()
            print("Moteur : dossier de travail sur DISQUE EXTERNE ->", str(p), flush=True)
            return p
        except Exception as e:
            print("Dossier externe indisponible (", ext, ") -> repli local :", e, flush=True)
    return BASE / "jobs"
WORK = _pick_work_dir()
WORK.mkdir(parents=True, exist_ok=True)
# Version du MOTEUR : renvoyée à la fin du rendu pour vérifier que le serveur qui tourne est bien à jour
# (dessins en parallèle). Si le popup n'affiche PAS cette version, c'est qu'un ancien moteur est resté actif.
SERVER_VERSION = "s20-nocap"
_LAST_ENC = "?"   # dernier encodeur vidéo utilisé (h264_videotoolbox = GPU, libx264 = CPU)

app = FastAPI(title="Foot Session Tracker")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS = {}
VID = WORK / "_videos"; VID.mkdir(exist_ok=True)
VKEYS = {}   # vkey -> chemin de la vidéo en cache (upload une seule fois)

def _prune_video_cache(max_gb=20.0):
    """PLAFOND DU CACHE (façon cloud_media_cache de LiveTag) : garde le dossier _videos sous une taille max.
    - supprime d'abord les uploads interrompus (.part) de plus de 2 h ;
    - puis, si le cache dépasse le plafond, enlève les COPIES les plus ANCIENNES (jamais touchées depuis >30 min)
      jusqu'à repasser sous le plafond. Les vidéos RÉFÉRENCÉES (lecture directe) vivent à leur vrai emplacement,
      PAS dans _videos -> elles ne sont jamais supprimées par ce nettoyage."""
    try:
        now = time.time()
        files = []
        for f in VID.glob("*"):
            try:
                if not f.is_file():
                    continue
                st = f.stat()
                # .part interrompu depuis plus de 2 h -> on jette
                if f.name.endswith(".part") and (now - st.st_mtime) > 7200:
                    try: f.unlink()
                    except Exception: pass
                    continue
                files.append((f, st.st_size, st.st_mtime))
            except Exception:
                pass
        total = sum(s for _, s, _ in files)
        cap = max_gb * (1024 ** 3)
        if total <= cap:
            return
        # du plus ANCIEN au plus récent ; on garde ceux touchés il y a < 30 min (peut-être en cours d'usage)
        files.sort(key=lambda x: x[2])
        for f, sz, mt in files:
            if total <= cap:
                break
            if (now - mt) < 1800:
                continue
            try:
                f.unlink()
                # on retire aussi les clés qui pointaient dessus
                for k, v in list(VKEYS.items()):
                    if v == str(f):
                        VKEYS.pop(k, None)
                total -= sz
                print("Cache moteur : fichier ancien supprimé (plafond) ->", f.name, flush=True)
            except Exception:
                pass
    except Exception as e:
        try: print("prune cache err:", e, flush=True)
        except Exception: pass

# ===== LICENCE / ACTIVATION (additif — INACTIF tant que LICENSE_SERVER_URL est vide) =====
import urllib.request as _ulr, urllib.parse as _ulp, platform as _plat
LICENSE_SERVER_URL = os.environ.get("LICENSE_SERVER_URL", "").rstrip("/")  # LOCAL: portail desactive (defaut vide). Client: definir la variable au lancement.
LICENSE_API_KEY = os.environ.get("LICENSE_API_KEY", "").strip()  # clé publiable Supabase (backend licence RPC)
# Config licence par FICHIER (licence.json dans le moteur) -> activable sans rebuild de l'app.
if not LICENSE_SERVER_URL:
    try:
        _lc = json.loads((BASE / "licence.json").read_text())
        LICENSE_SERVER_URL = (_lc.get("server_url") or "").rstrip("/")
        if not LICENSE_API_KEY:
            LICENSE_API_KEY = (_lc.get("api_key") or "").strip()
    except Exception:
        pass
_LIC_FILE = BASE / ".license.json"

def _machine_id():
    mid = None
    try:
        if _plat.system() == "Darwin":
            out = subprocess.check_output(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], text=True)
            m = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', out)
            if m: mid = m.group(1)
    except Exception:
        mid = None
    if not mid:
        try:
            if _LIC_FILE.exists(): mid = json.loads(_LIC_FILE.read_text()).get("mid")
        except Exception: mid = None
        if not mid: mid = uuid.uuid4().hex
    return mid

def _lic_load():
    try: return json.loads(_LIC_FILE.read_text())
    except Exception: return {}

def _lic_save(d):
    try:
        cur = _lic_load(); cur.update(d); _LIC_FILE.write_text(json.dumps(cur))
    except Exception: pass

import ssl as _ssl
def _lic_ctx():
    try:
        import certifi as _certifi
        return _ssl.create_default_context(cafile=_certifi.where())
    except Exception:
        try:
            return _ssl.create_default_context()
        except Exception:
            return None

def _lic_post(path, data):
    # Backend Supabase (RPC) : corps JSON + clé publiable dans les en-têtes.
    body = json.dumps(data).encode()
    req = _ulr.Request(LICENSE_SERVER_URL + path, data=body)
    req.add_header("Content-Type", "application/json")
    if LICENSE_API_KEY:
        req.add_header("apikey", LICENSE_API_KEY)
        req.add_header("Authorization", "Bearer " + LICENSE_API_KEY)
    try:
        with _ulr.urlopen(req, timeout=30, context=_lic_ctx()) as r:
            return json.loads(r.read().decode())
    except _ulr.HTTPError as e:
        # Réponse HTTP (404/402/409...) : on lit quand même le corps JSON du serveur.
        try:
            return json.loads(e.read().decode())
        except Exception:
            raise
    except Exception:
        # Repli macOS (certificats système non installés) : contexte non vérifié, TLS toujours chiffré.
        _unv = _ssl.create_default_context(); _unv.check_hostname = False; _unv.verify_mode = _ssl.CERT_NONE
        with _ulr.urlopen(req, timeout=30, context=_unv) as r:
            return json.loads(r.read().decode())

@app.get("/license/machine")
def license_machine():
    return {"machine_id": _machine_id(), "machine_name": _plat.node(), "configured": bool(LICENSE_SERVER_URL)}

@app.post("/license/activate")
def license_activate(code: str = Form(...)):
    if not LICENSE_SERVER_URL: return {"ok": True, "disabled": True}
    mid = _machine_id()
    try:
        r = _lic_post("/rpc/mlz_activate", {"p_code": code.strip().upper(), "p_machine_id": mid, "p_machine_name": _plat.node()})
    except Exception:
        return JSONResponse({"ok": False, "error": "serveur_injoignable"}, status_code=503)
    if r.get("ok"): _lic_save({"code": code.strip().upper(), "mid": mid})
    return r

@app.post("/license/trial")
def license_trial(email: str = Form(...)):
    if not LICENSE_SERVER_URL: return {"ok": True, "disabled": True}
    mid = _machine_id()
    try:
        r = _lic_post("/rpc/mlz_trial", {"p_email": email.strip(), "p_machine_id": mid, "p_machine_name": _plat.node()})
    except Exception:
        return JSONResponse({"ok": False, "error": "serveur_injoignable"}, status_code=503)
    if r.get("ok"):
        _lic_save({"code": r.get("code"), "mid": mid, "trial": True})
    return r

@app.get("/license/status")
def license_status():
    if not LICENSE_SERVER_URL: return {"ok": True, "disabled": True}
    code = _lic_load().get("code")
    if not code: return {"ok": False, "error": "non_active"}
    try:
        return _lic_post("/rpc/mlz_check", {"p_code": code, "p_machine_id": _machine_id()})
    except Exception:
        # tolérance hors-ligne : déjà activé -> on laisse travailler (l'annulation sera vue au retour du réseau)
        return {"ok": True, "offline": True}

@app.post("/license/release")
def license_release():
    if not LICENSE_SERVER_URL: return {"ok": True, "disabled": True}
    d = _lic_load(); code = d.get("code")
    if code:
        try: _lic_post("/rpc/mlz_release", {"p_code": code, "p_machine_id": _machine_id()})
        except Exception: pass
    _lic_save({"code": None}); return {"ok": True}

def cut_segment(src, t0, dur, outp):
    """Découpe rapide [t0, t0+dur] avec ffmpeg (seek rapide + ré-encodage léger)."""
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = "ffmpeg"
    cmd = [ff, "-y", "-ss", f"{t0:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
           "-an", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(outp)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=600)
    except Exception:
        return None
    return str(outp) if (os.path.exists(outp) and os.path.getsize(outp) > 0) else None

def _in_poly(x, y, poly):
    if not poly or len(poly) < 3: return True
    inside = False; n = len(poly); j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside
def _inflate_poly(poly, m=1.15):
    # Agrandit le polygone autour de son centre (marge de sécurité pour ne pas couper les joueurs au bord).
    if not poly or len(poly) < 3 or m == 1.0:
        return poly
    cx = sum(p[0] for p in poly) / len(poly); cy = sum(p[1] for p in poly) / len(poly)
    return [[cx + (p[0] - cx) * m, cy + (p[1] - cy) * m] for p in poly]
def _parse_poly(s):
    try:
        import json as _j
        p = _j.loads(s) if s else None
        if p and len(p) >= 3:
            return [[float(a[0]), float(a[1])] for a in p]
    except Exception:
        pass
    return None
def _parse_cals(s):
    # liste [{t, poly:[[x,y],...]}, ...] triée par t (calibrations multi-plans)
    try:
        import json as _j
        arr = _j.loads(s) if s else None
        out = []
        for c in (arr or []):
            pg = c.get("poly")
            if pg and len(pg) >= 3:
                out.append({"t": float(c.get("t", 0.0)),
                            "poly": [[float(a[0]), float(a[1])] for a in pg]})
        out.sort(key=lambda c: c["t"])
        return out or None
    except Exception:
        return None
def _poly_at(cals, t):
    # polygone du dernier point-clé dont t<=instant (sinon le plus proche)
    if not cals:
        return None
    best = None
    for c in cals:
        if c["t"] <= t + 1e-6 and (best is None or c["t"] > best["t"]):
            best = c
    if best is None:
        best = min(cals, key=lambda c: abs(c["t"] - t))
    return best["poly"]
_MODELS = {}
def get_model(name):
    if name not in _MODELS:
        from ultralytics import YOLO
        _MODELS[name] = YOLO(name)
    return _MODELS[name]

# ---- DETECTION JOUEURS (suivi) — SEPARE du calibrage (ne touche pas calibrate_smart/propagate) ----
_PLAYER_MODEL = {}
def get_player_model():
    if "m" not in _PLAYER_MODEL:
        from ultralytics import YOLO
        mdl = None
        # 1) CoreML (Neural Engine) — chaque .mlpackage est figé à sa taille d'export.
        #    yolov8x exporté à 1280 = le plus précis ; replis 640 si seuls m/n existent.
        # Le détecteur SPÉCIALISÉ FOOT (classes 0:ballon 1:gardien 2:joueur 3:arbitre, @1280) d'abord :
        # entraîné foot + haute résolution -> détecte bien plus de joueurs. Puis replis COCO (classe 0).
        # _pcls = indices des classes "joueur" à garder pour CE modèle.
        for c, sz, pc in (("football-player-detection.mlpackage", 1280, [1, 2, 3]),
                          ("yolov8x.mlpackage", 1280, [0]),
                          ("yolov8m.mlpackage", 640, [0]),
                          ("yolov8n.mlpackage", 640, [0])):
            if os.path.exists(c):
                try:
                    mdl = YOLO(c, task="detect"); mdl._coreml = True; mdl._imgsz = sz; mdl._pcls = pc
                    print("Suivi joueur : CoreML (Neural Engine) ->", c, "@", sz, "classes joueur", pc); break
                except Exception as e:
                    print("CoreML chargement echec (", c, "), repli :", e); mdl = None
        # 2) repli PyTorch sur GPU (MPS) — yolov8x à 1280 si pas encore converti en CoreML.
        if mdl is None:
            for p, sz in (("yolov8x.pt", 1280), ("yolov8m.pt", 1280), ("yolo11s.pt", 1280)):
                try:
                    mdl = YOLO(p); mdl._coreml = False; mdl._imgsz = sz; mdl._pcls = [0]
                    print("Suivi joueur : PyTorch ->", p, "@", sz, "(", pick_device(), ")"); break
                except Exception as e:
                    print("PT chargement echec (", p, "):", e); mdl = None
        _PLAYER_MODEL["m"] = mdl
    return _PLAYER_MODEL["m"]
def _pred_kw(model, imgsz, device):
    sz = getattr(model, "_imgsz", None) or imgsz
    if getattr(model, "_coreml", False):
        return dict(imgsz=sz)
    return dict(imgsz=sz, device=device)

# ---- Calibration IA : modèle YOLOv8-pose de détection du terrain (32 points-clés) ----
# Poids Roboflow "football-pitch-detection" (YOLOv8x-pose). Ordre des points-clés =
# SoccerPitchConfiguration.vertices, ici exprimés dans le repère terrain 105x68 (mètres).
_FIELD_MODEL = None
FIELD_KP_PATH = BASE / "football-pitch-detection.pt"
FIELD_KP_GID = "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"
FIELD_KP_MAP = [
    (0, 0), (0, 13.84), (0, 24.84), (0, 43.16), (0, 54.16), (0, 68),
    (5.5, 24.84), (5.5, 43.16), (11, 34),
    (16.5, 13.84), (16.5, 24.84), (16.5, 43.16), (16.5, 54.16),
    (52.5, 0), (52.5, 24.85), (52.5, 43.15), (52.5, 68),
    (88.5, 13.84), (88.5, 24.84), (88.5, 43.16), (88.5, 54.16),
    (94, 34), (99.5, 24.84), (99.5, 43.16),
    (105, 0), (105, 13.84), (105, 24.84), (105, 43.16), (105, 54.16), (105, 68),
    (43.35, 34), (61.65, 34),
]
def get_field_model():
    global _FIELD_MODEL
    if _FIELD_MODEL is not None:
        return _FIELD_MODEL
    # NEURAL ENGINE (CoreML) en priorité : accélère le placement des ancres (mêmes poids -> même précision).
    # Repli automatique sur PyTorch si le .mlpackage n'existe pas ou se charge mal.
    try:
        _ML = BASE / "football-pitch-detection.mlpackage"
        if _ML.exists():
            from ultralytics import YOLO
            m = YOLO(str(_ML), task="pose"); m._coreml = True; m._imgsz = 1280
            _FIELD_MODEL = m
            print("Calibration terrain : CoreML (Neural Engine) ->", _ML.name, flush=True)
            return _FIELD_MODEL
    except Exception as e:
        print("Terrain CoreML indisponible -> repli PyTorch :", e, flush=True)
    if not FIELD_KP_PATH.exists():
        try:
            import gdown
            gdown.download(id=FIELD_KP_GID, output=str(FIELD_KP_PATH), quiet=False)
        except Exception as e:
            print("Téléchargement modèle terrain IA échoué:", e)
            return None
    if not FIELD_KP_PATH.exists():
        return None
    try:
        from ultralytics import YOLO
        _FIELD_MODEL = YOLO(str(FIELD_KP_PATH))
    except Exception as e:
        print("Chargement modèle terrain IA échoué:", e)
        return None
    return _FIELD_MODEL
def ai_field_pts(im, conf=0.4, min_inliers=15):
    """Détecte les points-clés du terrain, PUIS valide géométriquement par homographie
    RANSAC : ne garde que les points cohérents (inliers). Renvoie None si le terrain
    n'est pas reconnu de façon fiable (l'app passe alors en correction manuelle)."""
    m = get_field_model()
    if m is None:
        return None
    try:
        import numpy as np, cv2
        H, W = im.shape[:2]
        _kw = dict(conf=0.2, imgsz=getattr(m, "_imgsz", 1280), verbose=False)
        if not getattr(m, "_coreml", False):
            _kw["device"] = pick_device()
        r = m.predict(im, **_kw)[0]
        kp = getattr(r, "keypoints", None)
        if kp is None or kp.xy is None or len(kp.xy) == 0:
            return None
        xy = kp.xy.cpu().numpy()
        cf = kp.conf.cpu().numpy() if getattr(kp, "conf", None) is not None else None
        # instance (terrain) avec le plus de points-clés fiables
        bi, bn = 0, -1
        for i in range(xy.shape[0]):
            n = int((cf[i] > conf).sum()) if cf is not None else xy.shape[1]
            if n > bn:
                bn, bi = n, i
        field, img = [], []
        for k in range(min(len(FIELD_KP_MAP), xy.shape[1])):
            c = float(cf[bi][k]) if cf is not None else 1.0
            if c < conf:
                continue
            x, y = float(xy[bi][k][0]), float(xy[bi][k][1])
            if x <= 1 and y <= 1:
                continue
            field.append(FIELD_KP_MAP[k]); img.append([x, y])
        if len(field) < min_inliers:
            return None
        F = np.array(field, dtype="float32"); I = np.array(img, dtype="float32")
        Hm, mask = cv2.findHomography(F, I, cv2.RANSAC, 6.0)
        if Hm is None or mask is None:
            return None
        inl = mask.ravel().astype(bool); n_inl = int(inl.sum())
        if n_inl < min_inliers:
            return None
        pts = [{"f": [float(F[j][0]), float(F[j][1])],
                "i": [round(float(I[j][0]) / W, 4), round(min(1.0, float(I[j][1]) / H), 4)], "c": 1.0}
               for j in range(len(field)) if inl[j]]
        return pts if len(pts) >= min_inliers else None
    except Exception as e:
        print("ai_field_pts err:", e)
        return None

def pick_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

# ---- EMPREINTE PROFONDE (ReID) — signature apparence apprise, bien plus fiable que la couleur ----
_REID = {}
OSNET_W = BASE / "osnet_x1_0_msmt17.pth"
OSNET_URL = ("https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
             "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_"
             "softmax_labelsmooth_flip_jitter.pth")
def get_reid():
    """Empreinte "signature" d'un joueur. EN PRIORITÉ : OSNet, un VRAI modèle de ré-identification de
    personnes (entraîné MSMT17) -> deux images du même joueur = vecteurs proches, joueurs différents =
    éloignés, bien plus discriminant qu'un CNN généraliste. REPLI automatique sur MobileNetV3-small si
    OSNet indisponible (pas de réseau, GPU incompatible…) -> le suivi continue comme avant."""
    if "m" not in _REID:
        _REID["m"] = None
        # 0) OSNet CoreML (NEURAL ENGINE) — le plus rapide. Actif seulement si le .mlpackage
        #    a été généré (2-convertir-coreml.command). Sinon on passe au repli torch ci-dessous.
        try:
            OSNET_ML = BASE / "osnet_x1_0_msmt17.mlpackage"
            if OSNET_ML.exists():
                import coremltools as ct, numpy as _np
                mlm = ct.models.MLModel(str(OSNET_ML))
                _sp = mlm.get_spec()
                in_name = _sp.description.input[0].name
                out_name = _sp.description.output[0].name
                r = mlm.predict({in_name: _np.zeros((1, 3, 256, 128), dtype=_np.float32)})
                dim = int(_np.asarray(r[out_name]).reshape(-1).shape[0])
                if dim >= 128:
                    _REID["m"] = mlm; _REID["kind"] = "osnet-coreml"; _REID["in"] = in_name; _REID["out"] = out_name
                    print("ReID profond : OSNet CoreML (Neural Engine) ->", OSNET_ML.name, "dim", dim, flush=True)
                    return _REID["m"]
        except Exception as e:
            print("OSNet CoreML indisponible -> repli torch :", e, flush=True)
        # 1) OSNet (vrai ReID) --------------------------------------------------
        try:
            import torch, importlib.util, urllib.request
            spec = importlib.util.spec_from_file_location("reid_osnet", str(BASE / "reid_osnet.py"))
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            if not OSNET_W.exists():
                print("ReID OSNet : téléchargement des poids (~16 Mo, 1re fois)…", flush=True)
                urllib.request.urlretrieve(OSNET_URL, str(OSNET_W))
            net = mod.osnet_x1_0(num_classes=1000); net.eval()
            ck = torch.load(str(OSNET_W), map_location="cpu")
            sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
            clean = {}
            for k, v in sd.items():
                kk = k[7:] if k.startswith("module.") else k
                if kk.startswith("classifier."):   # couche d'identités (inutile en eval) -> ignorée
                    continue
                clean[kk] = v
            net.load_state_dict(clean, strict=False)
            dev = pick_device()
            try: net.to(dev)
            except Exception: dev = "cpu"; net.to("cpu")
            # test d'inférence : si le GPU (MPS) refuse une couche, on bascule CPU ; si CPU échoue -> repli
            try:
                with torch.no_grad(): net(torch.zeros(1, 3, 256, 128).to(dev))
            except Exception:
                dev = "cpu"; net.to("cpu")
                with torch.no_grad(): net(torch.zeros(1, 3, 256, 128))
            from torchvision import transforms as _T
            tf = _T.Compose([_T.Resize((256, 128)), _T.ToTensor(),
                             _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            _REID["m"] = net; _REID["dev"] = dev; _REID["tf"] = tf; _REID["kind"] = "osnet"
            print("ReID profond : OSNet x1.0 (MSMT17) ->", dev, flush=True)
            return _REID["m"]
        except Exception as e:
            print("OSNet indisponible -> repli MobileNet :", e, flush=True)
            try:
                if OSNET_W.exists() and OSNET_W.stat().st_size < 1_000_000:
                    OSNET_W.unlink()   # téléchargement partiel/corrompu -> on le supprime pour réessayer plus tard
            except Exception: pass
        # 2) REPLI : MobileNetV3-small (comportement d'origine) -----------------
        try:
            import torch
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            w = MobileNet_V3_Small_Weights.DEFAULT
            net = mobilenet_v3_small(weights=w); net.classifier = torch.nn.Identity(); net.eval()
            dev = pick_device()
            try: net.to(dev)
            except Exception: dev = "cpu"; net.to("cpu")
            _REID["m"] = net; _REID["dev"] = dev; _REID["tf"] = w.transforms(); _REID["kind"] = "mobilenet"
            print("ReID : mobilenet_v3_small (repli) ->", dev)
        except Exception as e:
            print("ReID indisponible (repli couleur seule) :", e); _REID["m"] = None
    return _REID.get("m")
def reid_embed(im, boxes):
    """boxes = [(cx,cy,w,h), ...] en pixels. Renvoie une liste de vecteurs L2-normalisés
    (cosinus = produit scalaire) ou None par boîte si extraction impossible."""
    out = [None] * len(boxes)
    net = get_reid()
    if net is None:
        return out
    try:
        # --- Chemin NEURAL ENGINE (OSNet CoreML) : prétraitement identique, inférence sur l'ANE ---
        if _REID.get("kind") == "osnet-coreml":
            import numpy as np, cv2
            mlm = _REID["m"]; in_name = _REID["in"]; out_name = _REID["out"]
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            for i, (cx, cy, w, h) in enumerate(boxes):
                x0 = int(max(0, cx - w / 2)); x1 = int(min(im.shape[1], cx + w / 2))
                y0 = int(max(0, cy - h / 2)); y1 = int(min(im.shape[0], cy + h / 2))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                rgb = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                arr = np.transpose(rgb, (2, 0, 1))            # HWC -> CHW
                arr = ((arr - mean) / std)[None].astype(np.float32)   # (1,3,256,128)
                try:
                    f = np.asarray(mlm.predict({in_name: arr})[out_name]).reshape(-1).astype(np.float32)
                    out[i] = f / (np.linalg.norm(f) + 1e-9)
                except Exception:
                    pass
            return out
        import torch, cv2
        from PIL import Image
        tf = _REID["tf"]; dev = _REID["dev"]
        crops, idx = [], []
        for i, (cx, cy, w, h) in enumerate(boxes):
            x0 = int(max(0, cx - w / 2)); x1 = int(min(im.shape[1], cx + w / 2))
            y0 = int(max(0, cy - h / 2)); y1 = int(min(im.shape[0], cy + h / 2))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            rgb = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            crops.append(tf(Image.fromarray(rgb))); idx.append(i)
        if not crops:
            return out
        x = torch.stack(crops).to(dev)
        with torch.no_grad():
            f = net(x)
            f = torch.nn.functional.normalize(f, dim=1).cpu().numpy()
        for j, i in enumerate(idx):
            out[i] = f[j]
        return out
    except Exception as e:
        print("reid_embed err:", e); return out

BOTSORT_REID = """\
tracker_type: botsort
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 60
match_thresh: 0.8
fuse_score: true
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: true
model: auto
"""

PALETTE = ["#00e08f","#ff5d5d","#ffce3a","#4da6ff","#c879ff","#ff9d3a",
           "#36e0d0","#f25fb0","#9be33a","#5d7bff","#ff7a7a","#ffd86b"]
def color_for(tid): return PALETTE[int(tid) % len(PALETTE)]


_FT_CACHE = {}
def _frame_times(src):
    """Horodatages RÉELS (s) de chaque image, dans l'ordre d'affichage — gère la fréquence VARIABLE
    (enregistrements d'écran .mov/.mp4). Lu via ffprobe (métadonnées seules, AUCUN ré-encodage).
    Sert à dater les pistes par leur vrai temps au lieu de numéro/fps (qui dérive sur VFR).
    Renvoie [] si indisponible -> l'appelant retombe sur numéro/fps (fichiers à fps constant : OK)."""
    try:
        key = (src, int(os.path.getmtime(src)))
    except Exception:
        key = (src, 0)
    if key in _FT_CACHE:
        return _FT_CACHE[key]
    times = []
    import subprocess, re as _re
    # 1) ffprobe (rapide, métadonnées). On essaie chaque ffprobe voisin d'un ffmpeg connu + miniforge + PATH.
    probes = []
    for ff in _ffmpeg_bins():
        if ff.endswith("ffmpeg"):
            probes.append(ff[:-6] + "ffprobe")
    probes.append(str(BASE / "miniforge" / "bin" / "ffprobe"))
    probes.append("ffprobe")
    for probe in probes:
        try:
            out = subprocess.run([probe, "-v", "error", "-select_streams", "v:0",
                                  "-show_entries", "frame=best_effort_timestamp_time",
                                  "-of", "csv=p=0", src],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 text=True, timeout=180).stdout
            for line in out.splitlines():
                line = line.strip().rstrip(",")
                if not line or line == "N/A":
                    continue
                try:
                    times.append(float(line))
                except Exception:
                    pass
            if times:
                break
        except Exception:
            times = []
            continue
    # 2) REPLI robuste : ffmpeg showinfo (fonctionne avec N'IMPORTE quel ffmpeg, même sans ffprobe).
    #    Décode et imprime pts_time de chaque image dans l'ordre d'affichage.
    if not times:
        for ff in _ffmpeg_bins():
            try:
                res = subprocess.run([ff, "-hide_banner", "-nostats", "-i", src,
                                      "-map", "0:v:0", "-vf", "showinfo", "-f", "null", "-"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     text=True, timeout=600).stderr
                for mm in _re.finditer(r"pts_time:([0-9.]+)", res):
                    try:
                        times.append(float(mm.group(1)))
                    except Exception:
                        pass
                if times:
                    break
            except Exception:
                times = []
                continue
    if times:
        times.sort()   # ordre d'affichage (PTS croissant) = ordre de lecture OpenCV/Ultralytics
    try:
        print("  frame_times : %d horodatages lus (%s)" % (len(times), "vrais PTS" if times else "AUCUN -> repli numero/fps"), flush=True)
    except Exception:
        pass
    _FT_CACHE[key] = times
    return times

def run_tracking(job_id, video_path, model_name, conf, min_len, vid_stride, t0s, t1s, imgsz=640, poly=None, maxp=26, cals=None):
    job = JOBS[job_id]
    try:
        import cv2
        from ultralytics import YOLO

        cap0 = cv2.VideoCapture(str(video_path))
        fps = cap0.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        H = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap0.release()
        job.update(fps=fps, w=W, h=H, frames=total, dur=(total/fps if fps else 0))

        dur_total = (total / fps) if fps else 0
        offset = 0.0
        src = str(video_path)
        t1eff = t1s if (t1s and t1s > 0) else dur_total
        need_cut = (t0s > 0.05) or (dur_total and t1eff < dur_total - 0.05)
        if need_cut and t1eff > t0s:
            job["msg"] = "Préparation de l'extrait (découpe rapide)…"
            outp = job["dir"] / "clip.mp4"
            cut = cut_segment(video_path, max(0.0, t0s), (t1eff - t0s), outp)
            if cut:
                src = cut; offset = max(0.0, t0s)
                c2 = cv2.VideoCapture(src)
                fps = c2.get(cv2.CAP_PROP_FPS) or fps
                W = int(c2.get(cv2.CAP_PROP_FRAME_WIDTH) or W)
                H = int(c2.get(cv2.CAP_PROP_FRAME_HEIGHT) or H)
                total = int(c2.get(cv2.CAP_PROP_FRAME_COUNT) or total)
                c2.release()
        start_f = 0
        end_f = total
        span = max(1, end_f - start_f)

        device = pick_device()
        job["msg"] = f"Chargement du modèle {model_name} sur {device.upper()}… (1er usage : téléchargement)"
        model = get_model(model_name)

        tracker_file = job["dir"] / "botsort_reid.yaml"
        tracker_file.write_text(BOTSORT_REID, encoding="utf-8")
        tracker_arg = str(tracker_file)

        job["state"] = "running"

        # Horodatages réels des images (gère la fréquence variable) -> pistes datées au VRAI temps,
        # même base que le suivi seedé et l'affichage : le dessin reste collé au rectangle à tout instant.
        ftimes = _frame_times(src)
        # Pipeline NATIF Ultralytics (le moteur qui marche bien) : on streame la vidéo
        # dans l'ordre et on ne RETIENT que les images de la plage [start_f, end_f].
        def run_loop(tracker_name):
            tracks = {}
            perframe = []
            ndet = 0
            gen = model.track(source=src, stream=True, persist=True,
                              conf=conf, iou=0.5, classes=getattr(model, "_pcls", [0]), tracker=tracker_name,
                              device=device, vid_stride=vid_stride, imgsz=imgsz, verbose=False)
            fi = 0
            for r in gen:
                if JOBS.get(job_id, {}).get("cancel"):
                    break
                frame_no = fi * vid_stride
                if frame_no < start_f:
                    fi += 1
                    continue
                if end_f and frame_no > end_f:
                    break
                b = r.boxes
                # temps réel de CETTE image (utilisé pour les pistes ET les détections image-par-image)
                t = ((ftimes[frame_no] if (ftimes and 0 <= frame_no < len(ftimes)) else (frame_no / fps)) + offset)
                if b is not None and b.id is not None:
                    ids = b.id.int().cpu().tolist()
                    xywh = b.xywh.cpu().tolist()
                    cfs = b.conf.cpu().tolist()
                    ndet += len(ids)
                    for (cx, cy, w, h), tid, c in zip(xywh, ids, cfs):
                        tracks.setdefault(tid, []).append(
                            {"t": round(t, 3), "x": round(cx / W, 4),
                             "y": round(min(1.0, (cy + h/2) / H), 4),
                             "w": round(w / W, 4), "h": round(h / H, 4), "c": round(c, 2)})
                # DÉTECTIONS IMAGE PAR IMAGE : on garde TOUTES les boîtes de la frame (même sans ID de suivi),
                # filtrées « sur le terrain » -> l'affichage montre TOUS les joueurs présents à l'instant, même
                # si le suiveur n'a pas réussi à relier leur piste (cas vue haute : suivi qui fragmente).
                if b is not None and getattr(b, "xywh", None) is not None and len(b.xywh):
                    # AUCUN filtre terrain ici : on renvoie TOUTES les détections de la frame pour l'affichage
                    # (l'utilisateur voit tous les joueurs et choisit lui-même). Garantit qu'aucun joueur n'est
                    # masqué par un polygone de calibrage manuel imparfait.
                    fb = []
                    for (cx, cy, w, h) in b.xywh.cpu().tolist():
                        fx = cx / W; fy = min(1.0, (cy + h/2) / H)
                        fb.append({"x": round(fx, 4), "y": round(fy, 4),
                                   "w": round(w / W, 4), "h": round(h / H, 4)})
                    if fb:
                        perframe.append({"t": round(t, 3), "b": fb})
                job["progress"] = min(99, int((frame_no - start_f) / span * 100))
                job["msg"] = f"Analyse… {job['progress']}%  —  {len(tracks)} joueurs suivis"
                fi += 1
            job["_ndet"] = ndet
            job["_perframe"] = perframe
            return tracks

        try:
            tracks = run_loop(tracker_arg)
        except Exception:
            job["msg"] = "ReID indisponible — repli BoT-SORT standard…"
            tracks = run_loop("botsort.yaml")

        clean = []
        for tid, pts in tracks.items():
            if len(pts) >= min_len:
                pts.sort(key=lambda p: p["t"])
                clean.append({"id": int(tid), "n": len(pts), "color": color_for(tid), "points": pts})
        # 1) ne garder que les pistes majoritairement SUR le terrain (public/staff exclus)
        #    -> poly choisi PAR FRAME via les calibrations multi-plans (cals), sinon poly unique.
        if cals or poly:
            def _ratio(tr):
                ps = tr["points"]; ins = 0
                for p in ps:
                    pg = _poly_at(cals, p["t"]) if cals else poly
                    if pg is None or _in_poly(p["x"], p["y"], _inflate_poly(pg, 1.18)):
                        ins += 1
                return ins / max(1, len(ps))
            kept = [t for t in clean if _ratio(t) >= 0.35]
            # GARDE-FOU : on ne fait confiance au filtre terrain que s'il garde un nombre RAISONNABLE de joueurs.
            # Sinon (calibrage manuel imparfait -> polygone qui exclut presque tout le monde, cas « 1 seul joueur »),
            # on IGNORE le filtre et on garde toutes les pistes. Mieux vaut quelques personnes en trop que tout perdre.
            if len(kept) >= max(6, len(clean) // 3):
                clean = kept
        # 2) plafond : 22 joueurs + arbitre(s) (≈23) — on garde les pistes les plus longues
        clean.sort(key=lambda tr: -tr["n"])
        if maxp and len(clean) > maxp:
            clean = clean[:maxp]

        out = {"video": job["name"], "fps": fps, "width": W, "height": H,
               "frames": total, "duration": (total/fps if fps else 0),
               "range": [round(start_f/fps, 2), round(end_f/fps, 2)],
               "device": device, "model": model_name, "tracks": clean,
               "perframe": job.get("_perframe", [])}
        (job["dir"] / "tracks.json").write_text(json.dumps(out), encoding="utf-8")

        job["progress"] = 100
        job["state"] = "done"
        if clean:
            job["msg"] = f"Terminé ✅  {len(clean)} joueurs suivis · {len(job.get('_perframe', []))} images détectées."
        else:
            job["msg"] = ("Terminé, mais 0 joueur retenu. Détections brutes: %d. "
                          "Essaie une séquence avec des joueurs bien visibles, baisse 'conf' (0.2), "
                          "ou réduis 'durée mini'." % job.get("_ndet", 0))
    except Exception as e:
        job["state"] = "error"
        job["msg"] = "Erreur : " + str(e)
        job["trace"] = traceback.format_exc()
        print(job["trace"])


# ===================== PnLCalib (calibration IA forte, SoccerNet) =====================
PNL_DIR = BASE / "pnlcalib"
PNL_RAW = "https://raw.githubusercontent.com/mguti97/PnLCalib/main/"
PNL_FILES = ["model/cls_hrnet.py", "model/cls_hrnet_l.py", "utils/utils_calib.py",
             "utils/utils_heatmap.py", "utils/utils_optimize.py",
             "config/hrnetv2_w48.yaml", "config/hrnetv2_w48_l.yaml"]
PNL_KP_W = PNL_DIR / "SV_kp"
PNL_LINES_W = PNL_DIR / "SV_lines"
_PNL = None  # None=inconnu, False=indispo, dict=chargé
_PNL_ERR = None  # dernière erreur de reprojection (px) d'une calibration PnL

def ensure_pnl_files():
    for rel in PNL_FILES:
        p = PNL_DIR / rel; p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() or p.stat().st_size < 50:
            if not _dl(PNL_RAW + rel, p, min_size=50):
                print("PnL: échec téléchargement", rel); return False
    def big(url, path):
        if path.exists() and path.stat().st_size > 2e8:  # déjà ~253 Mo
            return True
        try:
            import torch
            torch.hub.download_url_to_file(url, str(path), progress=True)
        except Exception as e:
            print("PnL poids (torch) échec, repli urllib:", e)
            _dl(url, path, min_size=int(2e8))
        return path.exists() and path.stat().st_size > 2e8
    okk = big("https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp", PNL_KP_W)
    okl = big("https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines", PNL_LINES_W)
    return okk and okl

def get_pnl():
    global _PNL
    if _PNL is not None: return _PNL
    try:
        import sys, yaml, torch
        if not ensure_pnl_files(): _PNL = False; return False
        if str(PNL_DIR) not in sys.path: sys.path.insert(0, str(PNL_DIR))
        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l
        from utils.utils_calib import FramebyFrameCalib, keypoint_world_coords_2D
        from utils.utils_heatmap import (get_keypoints_from_heatmap_batch_maxpool,
                                         get_keypoints_from_heatmap_batch_maxpool_l,
                                         complete_keypoints, coords_to_dict)
        dev = pick_device()
        cfg = yaml.safe_load(open(PNL_DIR / "config/hrnetv2_w48.yaml"))
        cfg_l = yaml.safe_load(open(PNL_DIR / "config/hrnetv2_w48_l.yaml"))
        m = get_cls_net(cfg); m.load_state_dict(torch.load(str(PNL_KP_W), map_location=dev)); m.to(dev).eval()
        ml = get_cls_net_l(cfg_l); ml.load_state_dict(torch.load(str(PNL_LINES_W), map_location=dev)); ml.to(dev).eval()
        _PNL = dict(m=m, ml=ml, dev=dev, Cal=FramebyFrameCalib, kw=keypoint_world_coords_2D,
                    kpf=get_keypoints_from_heatmap_batch_maxpool, kpf_l=get_keypoints_from_heatmap_batch_maxpool_l,
                    complete=complete_keypoints, c2d=coords_to_dict)
        print("PnLCalib chargé ✅")
        return _PNL
    except Exception as e:
        import traceback; traceback.print_exc(); _PNL = False; return False

def _green_mask(im):
    """Masque de la pelouse (vert), dilaté pour inclure lignes/bords. Renvoie None si le vert ne
    couvre pas assez l'image (pour ne pas filtrer à tort sur une frame douteuse)."""
    import cv2, numpy as np
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (30, 25, 25), (95, 255, 255))
    if int(m.sum() // 255) < 0.12 * im.shape[0] * im.shape[1]:
        return None
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    k = max(8, int(0.035 * im.shape[0]))
    m = cv2.dilate(m, np.ones((k, k), np.uint8))
    return m

_PITCH_PTS = None
def _pitch_line_pts():
    """Points échantillonnés le long des lignes du MODÈLE de terrain (mètres, 105x68)."""
    import numpy as np, math
    pts = []
    def seg(p, q, nn=16):
        for s in range(nn + 1):
            a = s / nn; pts.append((p[0] + (q[0] - p[0]) * a, p[1] + (q[1] - p[1]) * a))
    seg((0, 0), (105, 0)); seg((105, 0), (105, 68)); seg((105, 68), (0, 68)); seg((0, 68), (0, 0))
    seg((52.5, 0), (52.5, 68))
    for k in range(40):
        a = 2 * math.pi * k / 40; pts.append((52.5 + 9.15 * math.cos(a), 34 + 9.15 * math.sin(a)))
    seg((0, 13.84), (16.5, 13.84)); seg((16.5, 13.84), (16.5, 54.16)); seg((16.5, 54.16), (0, 54.16))
    seg((105, 13.84), (88.5, 13.84)); seg((88.5, 13.84), (88.5, 54.16)); seg((88.5, 54.16), (105, 54.16))
    seg((0, 24.84), (5.5, 24.84)); seg((5.5, 24.84), (5.5, 43.16)); seg((5.5, 43.16), (0, 43.16))
    seg((105, 24.84), (99.5, 24.84)); seg((99.5, 24.84), (99.5, 43.16)); seg((99.5, 43.16), (105, 43.16))
    return np.array(pts, dtype="float32")

def line_overlap_score(im, corners_norm):
    """SCORE DE QUALITÉ RÉEL : fraction des points des lignes du modèle qui tombent sur du marquage
    blanc réellement présent (dans le vert). corners_norm = 4 coins terrain normalisés [0,1] dans
    l'ordre (0,0)(105,0)(105,68)(0,68). Renvoie None si trop peu de lignes sont dans le cadre."""
    import cv2, numpy as np
    global _PITCH_PTS
    if _PITCH_PTS is None: _PITCH_PTS = _pitch_line_pts()
    H0, W0 = im.shape[:2]
    sc = 960.0 / W0 if W0 > 960 else 1.0
    small = cv2.resize(im, (int(W0 * sc), int(H0 * sc))) if sc != 1.0 else im
    Hs2, Ws2 = small.shape[:2]
    src = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype="float32")
    dst = np.array([[c[0] * Ws2, c[1] * Hs2] for c in corners_norm], dtype="float32")
    try: Hf = cv2.getPerspectiveTransform(src, dst)
    except Exception: return None
    proj = cv2.perspectiveTransform(_PITCH_PTS.reshape(-1, 1, 2), Hf).reshape(-1, 2)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    white = (th > 18).astype(np.uint8)
    gm = _green_mask(small)
    if gm is not None: white = cv2.bitwise_and(white, (gm > 0).astype(np.uint8))
    white = cv2.dilate(white, np.ones((7, 7), np.uint8))  # tolérance ~±3 px
    inb = 0; hit = 0
    for (x, y) in proj:
        xi = int(round(x)); yi = int(round(y))
        if xi < 0 or yi < 0 or xi >= Ws2 or yi >= Hs2: continue
        inb += 1
        if white[yi, xi] > 0: hit += 1
    if inb < 20: return None
    return hit / inb

def _suppress_blue(im):
    """VEO / terrains à DOUBLE tracé : efface la peinture BLEUE (2e terrain peint sur le
    synthétique) en la remplaçant par du gazon, pour que PnLCalib et le recalage lignes ne
    voient qu'UN SEUL tracé blanc (sinon les intersections bleues parasites font échouer le
    calage). Borné à la PELOUSE (plus grande zone verte) -> ne touche jamais ciel/arbres/joueurs.
    Critère bleu robuste : le bleu domine nettement le rouge (B-R>12) alors que le blanc a R≈B."""
    import cv2, numpy as np
    try:
        h, w = im.shape[:2]
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (30, 18, 18), (95, 255, 255))
        gm = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats((gm > 0).astype(np.uint8), 8)
        if n <= 1:
            return im
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        pitch = (lab == big).astype(np.uint8)
        # REMPLIR le contour du terrain : les lignes (bleues/blanches) et les joueurs créent des
        # trous DANS la pelouse -> sans remplissage, le bleu au milieu d'une ligne n'est pas couvert.
        # On remplit le plus grand contour (bord = horizon) : trous bouchés, ciel/arbres exclus.
        try:
            cnts, _ = cv2.findContours(pitch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cbig = max(cnts, key=cv2.contourArea)
                filled = np.zeros_like(pitch); cv2.drawContours(filled, [cbig], -1, 1, -1)
                pitch = filled
        except Exception:
            pass
        pitch = cv2.dilate(pitch, np.ones((7, 7), np.uint8))
        b = im[:, :, 0].astype(np.int16); g = im[:, :, 1].astype(np.int16); r = im[:, :, 2].astype(np.int16)
        blue = ((b - r > 12) & (b - g > -3) & (b > 60) & (g > 35)).astype(np.uint8)
        blue = cv2.bitwise_and(blue, pitch)                       # bleu UNIQUEMENT sur la pelouse
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.dilate(blue, np.ones((9, 9), np.uint8))
        mask = cv2.bitwise_and(mask, pitch)                       # ne jamais déborder hors pelouse
        if int(mask.sum()) < 20:
            return im
        turf = cv2.medianBlur(im, 23)                             # couleur de gazon locale
        out = im.copy(); out[mask > 0] = turf[mask > 0]
        out = cv2.inpaint(out, mask, 3, cv2.INPAINT_TELEA)        # lissage final
        return out
    except Exception:
        return im

def _white_dt(small):
    """Carte de distance aux lignes blanches du terrain (petite résolution).
    dt[y,x] = distance au pixel de marquage blanc le plus proche (0 sur une ligne)."""
    import cv2, numpy as np
    small = _suppress_blue(small)   # VEO double tracé : on retire le bleu avant de détecter le blanc
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    white = (th > 18).astype(np.uint8)
    gm = _green_mask(small)
    if gm is not None: white = cv2.bitwise_and(white, (gm > 0).astype(np.uint8))
    if int(white.sum()) < 50: return None
    inv = (white == 0).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)

def line_snap(dt, corners_px):
    """RECALAGE SUR LES LIGNES : ajuste (décalage + échelle en espace image) la projection du terrain
    pour la coller au mieux sur les lignes blanches réelles. Ne déforme pas (garde la perspective de
    l'ancre), pousse juste le terrain sur le blanc. Renvoie (nouveaux_coins_px, residu_px)."""
    import cv2, numpy as np
    global _PITCH_PTS
    if _PITCH_PTS is None: _PITCH_PTS = _pitch_line_pts()
    Hh, Ww = dt.shape[:2]
    src = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype="float32")
    dst = np.array(corners_px, dtype="float32")
    try: Hf = cv2.getPerspectiveTransform(src, dst)
    except Exception: return corners_px, None
    P = cv2.perspectiveTransform(_PITCH_PTS.reshape(-1, 1, 2), Hf).reshape(-1, 2)
    inb = (P[:, 0] >= 0) & (P[:, 0] < Ww) & (P[:, 1] >= 0) & (P[:, 1] < Hh)
    P = P[inb]
    if len(P) < 25: return corners_px, None
    cen = P.mean(axis=0)
    # RECALAGE RIGIDE (similitude) : translation + ROTATION + zoom UNIFORME seulement.
    # Le terrain garde ses proportions réelles (jamais étiré d'un côté), on le glisse/pivote/
    # zoome juste pour tomber sur les lignes blanches. La rotation redresse la médiane en biais.
    def _R(th):
        c = np.cos(th); s = np.sin(th); return np.array([[c, -s], [s, c]])
    def cost(dx, dy, s, th):
        Q = cen + s * ((P - cen) @ _R(th).T) + np.array([dx, dy])
        xi = np.clip(Q[:, 0].astype(int), 0, Ww - 1); yi = np.clip(Q[:, 1].astype(int), 0, Hh - 1)
        return float(dt[yi, xi].mean())
    best = (0.0, 0.0, 1.0, 0.0); bc = cost(0, 0, 1.0, 0.0)
    # passe grossière (angles en radians ≈ ±1,7° / ±3,4°) puis fine
    for th in (-0.06, -0.03, 0.0, 0.03, 0.06):
        for s in (0.96, 1.0, 1.04):
            for dx in range(-28, 29, 7):
                for dy in range(-28, 29, 7):
                    c = cost(dx, dy, s, th)
                    if c < bc: bc = c; best = (dx, dy, s, th)
    bx, by, bs, bth = best
    for th in (bth - 0.02, bth - 0.01, bth, bth + 0.01, bth + 0.02):
        for s in (bs - 0.02, bs - 0.01, bs, bs + 0.01, bs + 0.02):
            for dx in range(int(bx) - 6, int(bx) + 7, 3):
                for dy in range(int(by) - 6, int(by) + 7, 3):
                    c = cost(dx, dy, s, th)
                    if c < bc: bc = c; best = (dx, dy, s, th)
    dx, dy, s, th = best; R = _R(th)
    new = [(cen + s * ((np.array(c, dtype="float64") - cen) @ R.T) + np.array([dx, dy])).tolist() for c in corners_px]
    return new, bc

def line_refine(dt, corners_px, iters=5):
    """AFFINAGE FIN (ICP) : à partir d'un placement grossier, ré-ajuste l'homographie COMPLÈTE
    (translation + rotation + perspective) pour coller les lignes du modèle au pixel blanc le plus
    proche, en quelques itérations. Réduit l'écart résiduel -> bien plus précis sur les lignes."""
    import cv2, numpy as np
    global _PITCH_PTS
    if _PITCH_PTS is None: _PITCH_PTS = _pitch_line_pts()
    Hh, Ww = dt.shape[:2]
    gy, gx = np.gradient(dt)  # gradient (lignes=y, colonnes=x) : pointe vers l'éloignement du blanc
    field_c = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype="float32")
    src = _PITCH_PTS
    cur = np.array(corners_px, dtype="float32")
    best = [list(map(float, c)) for c in corners_px]; best_res = None
    for _ in range(iters):
        try: Hf = cv2.getPerspectiveTransform(field_c, cur.astype("float32"))
        except Exception: break
        P = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hf).reshape(-1, 2)
        inb = (P[:, 0] >= 0) & (P[:, 0] < Ww) & (P[:, 1] >= 0) & (P[:, 1] < Hh)
        if int(inb.sum()) < 25: break
        xi = np.clip(P[:, 0].astype(int), 0, Ww - 1); yi = np.clip(P[:, 1].astype(int), 0, Hh - 1)
        d = dt[yi, xi]
        # RÉSIDU ROBUSTE : moyenne des 55% de points les PLUS PROCHES du blanc (= les lignes
        # réellement visibles). On ignore les points sans ligne (hors-cadre, effacées, masquées)
        # qui fausseraient le %. -> le % reflète l'alignement des lignes vraiment présentes.
        din = np.sort(d[inb]); kkeep = max(10, int(0.55 * len(din)))
        res = float(din[:kkeep].mean())
        if best_res is None or res < best_res:
            best_res = res; best = [list(map(float, c)) for c in cur]
        g = np.stack([gx[yi, xi], gy[yi, xi]], axis=1)
        gn = np.linalg.norm(g, axis=1, keepdims=True) + 1e-6
        T = P - (g / gn) * d[:, None]  # cible : chaque point poussé vers le blanc le plus proche
        Hf2, _ = cv2.findHomography(src[inb], T[inb], 0)  # ré-ajustement (moindres carrés)
        if Hf2 is None: break
        cur = cv2.perspectiveTransform(field_c.reshape(-1, 1, 2), Hf2.astype("float32")).reshape(-1, 2)
        # garde-fou : si ça part trop loin, on arrête
        if float(np.linalg.norm(cur - np.array(corners_px), axis=1).max()) > 0.25 * max(Ww, Hh): break
    return best, best_res

def _cdiag(s):
    """Diagnostic calibrage : imprime dans le Terminal ET écrit dans calib_diag.log (lisible après coup)."""
    try: print(s, flush=True)
    except Exception: pass
    try:
        with open(str(BASE / "calib_diag.log"), "a") as _f:
            import time as _t; _f.write(_t.strftime("%H:%M:%S ") + str(s) + "\n")
    except Exception: pass

def _json_safe(o):
    """Convertit récursivement tout objet (numpy float32/int64/arrays, etc.)
    en types Python natifs sérialisables en JSON. Évite les 500 silencieuses
    quand FastAPI ne sait pas encoder une valeur numpy du traitement 4K."""
    try:
        import numpy as _np
    except Exception:
        _np = None
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if _np is not None:
        if isinstance(o, _np.ndarray):
            return [_json_safe(v) for v in o.tolist()]
        if isinstance(o, _np.generic):
            o = o.item()
    if isinstance(o, float):
        # NaN / Inf ne sont pas du JSON valide -> on neutralise
        if o != o or o in (float("inf"), float("-inf")):
            return None
        return o
    if isinstance(o, (int, str, bool)) or o is None:
        return o
    try:
        return float(o)
    except Exception:
        return str(o)

def pnl_field_pts(im, kp_th=0.3434, line_th=0.7867, refine=True):
    """Calibration forte PnLCalib : image -> caméra -> 4 coins terrain (repère 105x68).
    refine=True : optimise l'homographie sur les lignes blanches (précis mais coûteux ~qq s).
    refine=False : homographie directe (rapide) — à utiliser quand une passe de recalage lignes
                   dédiée suit derrière (ancres), pour ne pas faire deux fois le même travail."""
    P = get_pnl()
    if not P: return None
    try:
        import torch, numpy as np, cv2
        import torchvision.transforms as T
        import torchvision.transforms.functional as f
        from PIL import Image
        dev = P["dev"]; H0, W0 = im.shape[:2]
        im = _suppress_blue(im)   # VEO double tracé : PnLCalib ne doit voir qu'un seul tracé blanc
        t = f.to_tensor(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))).float().unsqueeze(0)
        if t.shape[-1] != 960: t = T.Resize((540, 960))(t)
        t = t.to(dev); _, _, h, w = t.shape
        with torch.no_grad():
            hm = P["m"](t); hml = P["ml"](t)
        kp = P["c2d"](P["kpf"](hm[:, :-1, :, :]), threshold=kp_th)
        ln = P["c2d"](P["kpf_l"](hml[:, :-1, :, :]), threshold=line_th)
        # (Pas de filtrage par masque pelouse ici : il supprimait des détections valides et affaiblissait
        #  la reconnaissance. La confusion ligne/panneau est gérée en aval par la cohérence temporelle
        #  des ancres, qui n'enlève aucune détection.)
        kp_dict, lines_dict = P["complete"](kp[0], ln[0], w=w, h=h, normalize=True)
        try:
            _nk = sum(1 for _v in (kp_dict.values() if hasattr(kp_dict, "values") else []) if isinstance(_v, dict) and _v)
            _nl = sum(1 for _v in (lines_dict.values() if hasattr(lines_dict, "values") else []) if _v)
            _cdiag("    [PNL-DIAG] img=%dx%d  points-cles=%d  lignes=%d  (seuils kp=%.3f ligne=%.3f)" % (W0, H0, _nk, _nl, kp_th, line_th))
        except Exception as _de:
            _cdiag("    [PNL-DIAG] compteur err: %s" % _de)
        cam = P["Cal"](iwidth=W0, iheight=H0, denormalize=True)
        cam.update(kp_dict, lines_dict)
        # Homographie du SOL (terrain plat) + RAFFINEMENT PAR LES LIGNES : l'homographie est optimisée
        # pour que les lignes du modèle se superposent aux lignes blanches réellement détectées dans
        # l'image -> calibrage beaucoup plus net le long des lignes du terrain. Repli sans raffinement
        # si l'optimisation échoue sur une image (on ne perd pas l'ancre).
        try:
            res = cam.heuristic_voting_ground(refine_lines=refine)
            if (not res or res.get("homography") is None) and refine:
                res = cam.heuristic_voting_ground(refine_lines=False)
        except Exception:
            try:
                res = cam.heuristic_voting_ground(refine_lines=False)
            except Exception:
                res = None
        if not res or res.get("homography") is None:
            _cdiag("    [PNL-DIAG] -> VOTE CAMERA ECHOUE (pas d'homographie) : terrain non reconnu sur cette image")
            return None
        globals()["_PNL_ERR"] = float(res.get("rep_err") or 0.0)
        _cdiag("    [PNL-DIAG] -> OK homographie, err reproj=%.4f" % (globals().get("_PNL_ERR") or 0.0))
        Hi2f = np.array(res["homography"], dtype="float64")   # image(px) -> terrain (centré m)
        Hf2i = np.linalg.inv(Hi2f)                              # terrain (centré m) -> image(px)
        out = []
        for (fx, fy) in [(0, 0), (105, 0), (105, 68), (0, 68)]:
            ip = Hf2i @ np.array([fx - 52.5, fy - 34, 1.0]); _w = ip[-1]; ip = ip / (_w if abs(_w) > 1e-9 else 1e-9)   # garde anti division par zero (coin projeté à l'horizon)
            out.append({"f": [fx, fy], "i": [round(float(ip[0]) / W0, 4), round(float(ip[1]) / H0, 4)]})
        return out
    except Exception as e:
        import traceback; traceback.print_exc(); return None

def field_pts(im, minp=15, refine=True):
    """Calibration : PnLCalib (fort) en priorité, sinon repli YOLO-pose 32 points."""
    p = pnl_field_pts(im, refine=refine)
    if p and len(p) >= 4: return p
    return ai_field_pts(im, min_inliers=max(4, int(minp)))

def pnl_reproj_err(im, Hf):
    """Mesure le décalage RÉEL d'une homographie Hf (terrain 0..105 -> image normalisée) :
    re-détecte les points-clés du terrain et compare aux positions projetées. Renvoie l'erreur
    médiane normalisée (0 = parfait), ou None si pas assez de points détectés."""
    P = get_pnl()
    if not P:
        return None
    try:
        import torch, numpy as np, cv2
        import torchvision.transforms as T
        import torchvision.transforms.functional as f
        from PIL import Image
        t = f.to_tensor(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))).float().unsqueeze(0)
        if t.shape[-1] != 960:
            t = T.Resize((540, 960))(t)
        t = t.to(P["dev"]); _, _, h, w = t.shape
        with torch.no_grad():
            hm = P["m"](t)
        kpc = P["kpf"](hm[:, :-1, :, :])[0].cpu().numpy()  # (C,1,3) [x,y,score] px réseau
        kw = P["kw"]; errs = []
        for k in range(min(len(kw), kpc.shape[0])):
            if float(kpc[k, 0, 2]) < 0.5:
                continue
            dx = float(kpc[k, 0, 0]) / w; dy = float(kpc[k, 0, 1]) / h
            px, py = _aph(Hf, kw[k][0] + 52.5, kw[k][1] + 34.0)
            errs.append(((dx - px) ** 2 + (dy - py) ** 2) ** 0.5)
        if len(errs) < 4:
            return None
        errs.sort()
        return errs[len(errs) // 2]  # médiane (normalisée)
    except Exception as e:
        print("pnl_reproj_err err:", e)
        return None

NANO_BB = BASE / "nanotrack_backbone.onnx"
NANO_HEAD = BASE / "nanotrack_head.onnx"
def _dl(url, path, min_size=1000):
    try:
        import urllib.request, shutil
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        ok = path.exists() and path.stat().st_size > min_size
        if not ok:
            print("dl trop petit", url, path.stat().st_size if path.exists() else 0)
        return ok
    except Exception as e:
        print("dl fail", url, e)
        return False
_NANO_OK = None  # None=inconnu, False=indispo (ne plus réessayer)
def make_tracker(kind):
    """Crée un traqueur d'objet OpenCV : 'nano' (rapide, réseau) ou 'csrt' (précis)."""
    import cv2
    global _NANO_OK
    if kind == "nano" and _NANO_OK is not False:
        try:
            if not (NANO_BB.exists() and NANO_HEAD.exists()):
                for url in ("https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/object_tracking_nanotrack/object_tracking_nanotrack_backbone_2022nov.onnx",
                            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_tracking_nanotrack/object_tracking_nanotrack_backbone_2022nov.onnx"):
                    if _dl(url, NANO_BB): break
                for url in ("https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/object_tracking_nanotrack/object_tracking_nanotrack_head_2022nov.onnx",
                            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_tracking_nanotrack/object_tracking_nanotrack_head_2022nov.onnx"):
                    if _dl(url, NANO_HEAD): break
            if NANO_BB.exists() and NANO_HEAD.exists():
                p = cv2.TrackerNano_Params(); p.backbone = str(NANO_BB); p.neckhead = str(NANO_HEAD)
                _NANO_OK = True
                return cv2.TrackerNano_create(p)
            _NANO_OK = False  # téléchargement impossible -> on n'essaiera plus, repli CSRT
        except Exception as e:
            _NANO_OK = False
            print("nano indispo, repli CSRT:", e)
    for maker in ("TrackerCSRT_create",):
        try:
            return getattr(cv2, maker)()
        except Exception:
            pass
    try:
        return cv2.legacy.TrackerCSRT_create()
    except Exception:
        return None

_DET_CACHE = {}   # détections YOLO partagées entre joueurs d'un même clip : {(src, index_image): [(cx,cy,w,h),...]}
def run_track1(job_id, video_path, t0s, t1s, seedx, seedy, model_name, imgsz, conf, poly=None, cals=None, tracker="csrt", seed_bw=0.0, seed_bh=0.0, t0min=None):
    """Suivi MONO-joueur type LiveTag : un traqueur d'objet (CSRT/Nano) VERROUILLÉ sur le
    joueur cliqué, suivi image par image, avec re-détection périodique anti-dérive."""
    job = JOBS[job_id]
    try:
        import cv2
        c0 = cv2.VideoCapture(str(video_path)); fps = c0.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(c0.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); H = int(c0.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ntot = int(c0.get(cv2.CAP_PROP_FRAME_COUNT) or 0); c0.release()
        dur_total = (ntot / fps) if fps else 0
        # VITESSE : plus de ré-encodage du segment. On lit DIRECTEMENT dans la vidéo (seek),
        # on saute à l'instant de départ et on s'arrête à t1. Mêmes images d'origine = même qualité.
        offset = max(0.0, t0s); src = str(video_path)
        t1eff = t1s if (t1s and t1s > 0) else dur_total
        total = max(1, int(round((t1eff - offset) * fps)))   # nb d'images du segment (barre de progression)
        start_fi = int(round(offset * fps))
        device = pick_device(); job["state"] = "running"; job["msg"] = "Suivi IA…"
        det_model = get_player_model()

        def detect_boxes(im, t, absfi=None):
            # seuil abaissé (min 0.18) : capte les joueurs peu visibles/partiellement cachés qui étaient
            # manqués. Les fausses détections éventuelles sont écartées par le portail + le ReID.
            # VITESSE MULTI-JOUEURS : la détection d'une image est IDENTIQUE pour tous les joueurs du même
            # clip -> on la calcule une seule fois et on la réutilise (cache par index d'image partagé).
            raw = None
            ckey = (src, int(absfi)) if absfi is not None else None
            if ckey is not None:
                raw = _DET_CACHE.get(ckey)
            if raw is None:
                _cf = min(float(conf), 0.18)
                r = det_model.predict(im, conf=_cf, classes=getattr(det_model, "_pcls", [0]), verbose=False, **_pred_kw(det_model, imgsz, device))[0]
                raw = r.boxes.xywh.cpu().tolist() if (r.boxes is not None) else []
                if ckey is not None:
                    if len(_DET_CACHE) > 8000:
                        _DET_CACHE.clear()
                    _DET_CACHE[ckey] = raw
            pg = (_poly_at(cals, t) if cals else poly)
            out = []
            for (cx, cy, w, h) in raw:
                fx, fy = cx, cy + h / 2.0
                if pg and not _in_poly(fx / W, fy / H, pg):
                    continue
                out.append((cx, cy, w, h, fx, fy))
            return out
        def nearest(boxes, px, py):
            best, bd = None, 1e18
            for bx in boxes:
                d = (bx[4] - px) ** 2 + (bx[5] - py) ** 2
                if d < bd: bd = d; best = bx
            return best, bd

        # ===== MOTEUR "VERROU IA" : détection 1280 + EMPREINTE PROFONDE (ReID) + couleur =====
        #   + COMPENSATION DU MOUVEMENT CAMÉRA (GMC, flux optique) + anti-occlusion.
        # À chaque image : YOLO détecte tous les joueurs ; on choisit celui dont l'empreinte
        # profonde + la couleur collent au joueur verrouillé ET dont la position colle à la
        # prédiction APRÈS avoir retiré le déplacement de la caméra. Si personne ne correspond
        # (occlusion), on n'attrape PAS un autre joueur : on avance en aveugle et on réacquiert
        # par l'apparence dès que possible.
        def torso_hist(im, cx, cy, w, h):
            x0 = int(max(0, cx - w * 0.30)); x1 = int(min(im.shape[1], cx + w * 0.30))
            y0 = int(max(0, cy - h * 0.45)); y1 = int(min(im.shape[0], cy - h * 0.05))
            if x1 - x0 < 3 or y1 - y0 < 3:
                return None
            hsv = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))   # pelouse
            mask = cv2.bitwise_not(green)
            hist = cv2.calcHist([hsv], [0, 1], mask, [32, 32], [0, 180, 0, 256])
            if hist.sum() < 1:
                hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            if hist.sum() < 1:
                return None
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return hist
        def hist_sim(a, b):
            if a is None or b is None:
                return 0.5
            v = cv2.compareHist(a, b, cv2.HISTCMP_CORREL)
            return max(0.0, min(1.0, (v + 1.0) / 2.0))
        def cos_sim(a, b):
            if a is None or b is None:
                return 0.5
            return max(0.0, min(1.0, float((a * b).sum())))
        def cam_shift(prev_gray, gray):
            """Transformation globale prev->courant (mouvement/zoom CAMÉRA) par flux optique.
            VITESSE : le flux optique tourne sur une image RÉDUITE (~960 px) au lieu de la pleine
            résolution (4K = ~16x plus de pixels) -> beaucoup plus rapide, précision identique (on
            remet la translation à l'échelle pleine résolution à la fin)."""
            try:
                W0 = gray.shape[1]
                sc = 960.0 / W0 if W0 > 960 else 1.0
                pg = cv2.resize(prev_gray, None, fx=sc, fy=sc) if sc != 1.0 else prev_gray
                cg = cv2.resize(gray, None, fx=sc, fy=sc) if sc != 1.0 else gray
                p0 = cv2.goodFeaturesToTrack(pg, 200, 0.01, 30)
                if p0 is None or len(p0) < 12:
                    return None
                p1, st, _ = cv2.calcOpticalFlowPyrLK(pg, cg, p0, None)
                if p1 is None:
                    return None
                st = st.ravel().astype(bool); a0 = p0[st]; a1 = p1[st]
                if len(a0) < 12:
                    return None
                M, _ = cv2.estimateAffinePartial2D(a0, a1, method=cv2.RANSAC)
                if M is not None and sc != 1.0:
                    M = M.copy(); M[0, 2] /= sc; M[1, 2] /= sc   # translation -> échelle pleine résolution
                return M
            except Exception:
                return None
        def apply_M(M, x, y):
            if M is None:
                return x, y
            return (M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2])

        cap = cv2.VideoCapture(src)
        # --- Positionnement au début du segment (méthode "LiveTag") ---
        # Sur les fichiers à fréquence VARIABLE (enregistrements d'écran, WebM), l'index d'image ne
        # correspond PAS au temps réel : frame_n / fps est faux. On se cale sur le vrai horodatage
        # (PTS) de chaque image lu en avançant -> aucune conversion nécessaire, calage exact.
        def _cap_pts():
            v = cap.get(cv2.CAP_PROP_POS_MSEC)
            return (v / 1000.0) if (v and v > 1e-6) else None
        USE_PTS = False
        seg_t0 = offset
        if start_fi > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_fi)   # saut approximatif près du début du segment
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("vidéo illisible")
        _pts = _cap_pts()
        if _pts is not None:
            USE_PTS = True
            _fps_est = fps if fps else 25.0
            # a) on a DÉPASSÉ le début voulu (index trop grand sur VFR) -> on recule et on réessaie
            _tries = 0
            while _pts is not None and _pts > offset + 0.15 and start_fi > 0 and _tries < 6:
                start_fi = max(0, start_fi - (int((_pts - offset) * _fps_est) + 5))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_fi)
                ret, frame = cap.read()
                if not ret:
                    break
                _pts = _cap_pts(); _tries += 1
            # b) on est AVANT le début voulu -> on avance image par image jusqu'au bon temps réel
            _guard = 0
            while _pts is not None and _pts < offset - 0.05 and _guard < 200000:
                ret, frame = cap.read()
                if not ret:
                    break
                _n = _cap_pts()
                if _n is None:
                    break
                _pts = _n; _guard += 1
            if _pts is not None:
                seg_t0 = _pts
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sx, sy = seedx * W, seedy * H
        _dets = detect_boxes(frame, offset, start_fi)
        seed_box = None
        # 1) PRIORITÉ : la détection dont la BOÎTE CONTIENT le point cliqué -> on est posé dessus.
        #    Ça garantit que si l'utilisateur pose sur le gardien, c'est le gardien qui est verrouillé,
        #    même si un joueur de champ est un peu plus proche en distance pure.
        for b in _dets:
            bx, by, bw2, bh2 = b[0], b[1], b[2], b[3]
            if abs(sx - bx) <= bw2 * 0.55 and (by - bh2 * 0.55) <= sy <= (by + bh2 * 0.60):
                if seed_box is None or ((b[0] - sx) ** 2 + (b[1] - sy) ** 2) < ((seed_box[0] - sx) ** 2 + (seed_box[1] - sy) ** 2):
                    seed_box = b
        # 2) sinon la plus proche, mais SEULEMENT si vraiment sous le clic (sinon gardien/arbitre non
        #    détecté : on ne saute pas sur un voisin, on garde la personne désignée).
        if seed_box is None:
            nb, _sd = nearest(_dets, sx, sy)
            _seed_r = max(W * 0.02, (seed_bw * W * 0.6) if seed_bw > 0 else (W * 0.035))
            if nb is not None and (_sd ** 0.5) <= _seed_r:
                seed_box = nb
        if seed_box is not None:
            cx, cy, w, h, _, _ = seed_box
        else:
            w = (seed_bw * W) if seed_bw > 0 else (W * 0.045)
            h = (seed_bh * H) if seed_bh > 0 else (H * 0.13)
            cx, cy = sx, sy - h / 2
        tgt_hist = torso_hist(frame, cx, cy, w, h)
        tgt_emb = reid_embed(frame, [(cx, cy, w, h)])[0]              # empreinte profonde de référence
        tw, th = w, h
        # ===== FILTRE DE KALMAN (vitesse constante) : état s=[x, y, vx, vy], covariance P =====
        # Prédit la position ET son incertitude -> gate adaptatif, trajectoire lissée, meilleur
        # re-verrouillage après occlusion. Combiné à l'apparence OSNet pour l'identité.
        import numpy as np
        s = np.array([cx, cy, 0.0, 0.0], dtype="float64")
        P = np.diag([25.0, 25.0, 400.0, 400.0]).astype("float64")
        Fm = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype="float64")
        Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype="float64")
        Qm = np.diag([4.0, 4.0, 16.0, 16.0]).astype("float64")       # bruit de process
        Rm = np.diag([25.0, 25.0]).astype("float64")                 # bruit de mesure (px^2)
        I4 = np.eye(4)
        last_cx, last_cy = cx, cy
        diag = (W * W + H * H) ** 0.5
        pts = []
        def record(t, ccx, ccy, ww, hh, conf):
            fx = ccx / W; fy = min(1.0, (ccy + hh / 2) / H)
            pts.append({"t": round(t, 3), "x": round(fx, 4), "y": round(fy, 4),
                        "w": round(ww / W, 4), "h": round(hh / H, 4), "c": round(conf, 2)})
        record(seg_t0, cx, cy, w, h, 0.95)
        fi = 0; coast = 0
        STRIDE = max(1, int(round(fps / 10.0)))              # ~10 img/s analysées (le rendu interpole) -> plus rapide
        MAXCOAST = max(6, int(round(fps * 1.8 / STRIDE)))    # tolérance occlusion, en "pas" traités
        GATE_K = min(3.0, float(STRIDE))                     # le joueur bouge plus entre 2 images traitées
        # poids du coût : empreinte profonde (OSNet) TRÈS dominante maintenant qu'elle est fiable,
        # puis couleur, position, taille.
        W_EMB, W_COL, W_POS, W_SIZE = 2.2, 0.6, 1.0, 0.3
        # Traitement d'UNE image analysée — factorisé pour être rejoué en AVANT (départ->fin) ET en
        # ARRIÈRE (départ->début du clip), afin de calculer TOUJOURS le clip entier quel que soit
        # l'instant de départ choisi (précision identique à un suivi lancé depuis le début).
        def process(frame, t, absfi):
            nonlocal s, P, tw, th, tgt_hist, tgt_emb, last_cx, last_cy, coast, prev_gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            M = cam_shift(prev_gray, gray); prev_gray = gray
            # 1) compensation caméra : on déplace l'état (position + vitesse) par le mouvement caméra
            if M is not None:
                A = M[:, :2]; tvec = M[:, 2]
                s[:2] = A.dot(s[:2]) + tvec
                s[2:] = A.dot(s[2:])
            # 2) prédiction Kalman (vitesse constante) : s = F·s ; P = F·P·Fᵀ + Q (l'incertitude grandit)
            s = Fm.dot(s)
            P = Fm.dot(P).dot(Fm.T) + Qm
            px, py = float(s[0]), float(s[1])
            boxes = detect_boxes(frame, t, absfi)
            # MODE RÉACQUISITION : si le joueur est perdu depuis longtemps (occlusion/sortie de cadre),
            # on ouvre la fenêtre à presque toute l'image et on se fie à l'APPARENCE pour le retrouver
            # sans sauter sur un autre joueur.
            LOST = coast > MAXCOAST
            if LOST:
                gate = diag * 0.85
            else:
                # gate ADAPTATIF (Kalman) : grandit avec l'incertitude de position -> large quand on vient
                # de perdre le joueur, serré quand on le suit bien. Anti-saut + meilleure réacquisition.
                sig = float(np.sqrt(max(1.0, P[0, 0] + P[1, 1])))
                gate = min(diag * 0.5, diag * 0.07 * GATE_K + 2.2 * sig)
            # VITESSE : le ReID (OSNet, coûteux) est calculé UNIQUEMENT sur les détections proches de la
            # prédiction (ou toutes si perdu). Les joueurs lointains ne sont pas candidats -> inutile.
            cand = [bi for bi, b in enumerate(boxes)
                    if LOST or ((b[0] - px) ** 2 + (b[1] - py) ** 2) ** 0.5 <= gate * 2.0]
            embs = {}
            if cand:
                ce = reid_embed(frame, [(boxes[bi][0], boxes[bi][1], boxes[bi][2], boxes[bi][3]) for bi in cand])
                for k, bi in enumerate(cand):
                    embs[bi] = ce[k]
            best = None; best_cost = 1e9; best_q = 0.0; best_bi = -1; best_semb = 0.0
            for bi in cand:
                bcx, bcy, bw, bh, _, _ = boxes[bi]
                dpos = ((bcx - px) ** 2 + (bcy - py) ** 2) ** 0.5
                semb = cos_sim(tgt_emb, embs.get(bi))
                scol = hist_sim(tgt_hist, torso_hist(frame, bcx, bcy, bw, bh))
                dsize = min(1.0, abs(bw - tw) / max(1.0, tw))
                if LOST:
                    cost = 1.4 * (1.0 - semb) + 0.6 * (1.0 - scol)   # apparence seule
                else:
                    cost = (W_EMB * (1.0 - semb) + W_COL * (1.0 - scol)
                            + W_POS * (dpos / max(1.0, gate)) + W_SIZE * dsize)
                q = 0.7 * semb + 0.3 * scol                          # qualité d'apparence [0..1]
                if cost < best_cost:
                    best_cost = cost; best = (bcx, bcy, bw, bh); best_q = q; best_bi = bi; best_semb = semb
            if LOST:                                                 # réacquisition : apparence TRÈS forte exigée
                accept = (best is not None) and (best_q > 0.62) and (best_semb > 0.60)
            else:
                # ANTI-TRANSFERT (renforcé) : on refuse une détection dont l'apparence (OSNet) n'est pas
                # NETTEMENT proche. Un joueur d'allure différente (autre maillot, gardien vs joueur de champ)
                # est refusé -> JAMAIS de saut d'identité. En cas de doute on préfère perdre (trou) que sauter.
                accept = (best is not None) and (best_cost < 1.4) and (best_semb > 0.45) and (best_q > 0.40 or coast == 0)
            if accept:
                bcx, bcy, bw, bh = best
                # MISE À JOUR KALMAN avec la détection mesurée -> corrige position + vitesse, réduit P.
                z = np.array([float(bcx), float(bcy)])
                if LOST:                                             # réacquisition : on repositionne franchement
                    s = np.array([float(bcx), float(bcy), 0.0, 0.0]); P = np.diag([25.0, 25.0, 400.0, 400.0]).astype("float64")
                else:
                    yk = z - Hm.dot(s); Sk = Hm.dot(P).dot(Hm.T) + Rm
                    Kk = P.dot(Hm.T).dot(np.linalg.inv(Sk))
                    s = s + Kk.dot(yk); P = (I4 - Kk.dot(Hm)).dot(P)
                last_cx, last_cy = float(s[0]), float(s[1])
                tw = 0.8 * tw + 0.2 * bw; th = 0.8 * th + 0.2 * bh
                if best_q > 0.6:                                      # MAJ lente des deux signatures
                    nh = torso_hist(frame, bcx, bcy, bw, bh)
                    if nh is not None and tgt_hist is not None:
                        tgt_hist = cv2.addWeighted(tgt_hist, 0.9, nh, 0.1, 0)
                    ne = embs.get(best_bi) if best_bi >= 0 else None
                    if ne is not None and tgt_emb is not None:
                        mix = 0.9 * tgt_emb + 0.1 * ne; n = float((mix * mix).sum()) ** 0.5
                        if n > 1e-6: tgt_emb = mix / n
                coast = 0
                record(t, last_cx, last_cy, tw, th, min(0.99, best_q))   # position LISSÉE par le Kalman
            else:                                                    # PERDU : on NE raccroche PAS un autre joueur
                coast += 1
                s[2:] *= 0.85                                         # amortit la vitesse (pas de dérive)
                last_cx = min(max(0.0, float(s[0])), W); last_cy = min(max(0.0, float(s[1])), H)
                # petit pont pour un trou de 1-2 images (anti-clignotement). AU-DELÀ : on n'enregistre plus
                # rien -> TROU dans la piste -> le rectangle DISPARAÎT (le joueur est absent, sorti du cadre).
                # On continue de prédire pour re-verrouiller le BON joueur s'il revient (apparence forte),
                # sinon la piste reste trouée. => jamais de transfert sur un autre joueur.
                if coast <= 2:
                    record(t, last_cx, last_cy, tw, th, 0.2)

        # instantané de l'état APRÈS amorçage -> point de départ commun aux deux passes
        _seed_snap = dict(s=s.copy(), P=P.copy(), tw=tw, th=th,
                          hist=(None if tgt_hist is None else tgt_hist.copy()),
                          emb=(None if tgt_emb is None else tgt_emb.copy()),
                          gray=prev_gray.copy(), cx=last_cx, cy=last_cy)
        seed_fi = int(round(seg_t0 * fps)) if fps else start_fi
        # ===== PASSE AVANT : du point de départ jusqu'à la fin du segment =====
        while True:
            if JOBS.get(job_id, {}).get("cancel"): break
            for _ in range(STRIDE - 1):                   # saute (sans analyser) les images intermédiaires
                if not cap.grab(): break
                fi += 1
            ret, frame = cap.read()
            if not ret: break
            fi += 1
            if USE_PTS:
                _pm = cap.get(cv2.CAP_PROP_POS_MSEC)      # vrai horodatage de l'image (gère le VFR)
                t = (_pm / 1000.0) if (_pm and _pm > 1e-6) else (fi / fps + offset)
            else:
                t = fi / fps + offset
            if t1eff and t > t1eff + 1e-3: break          # fin du segment
            process(frame, t, start_fi + fi)
            if total: job["progress"] = min(98, int(fi / total * 100))
            job["msg"] = f"Suivi IA (ReID + caméra)… {job['progress']}%"
        # ===== PASSE ARRIÈRE : du point de départ jusqu'au DÉBUT du clip (calcul complet) =====
        # -> le clip entier est calculé même si tu démarres à 5s : élargir la fenêtre d'affichage
        #    ensuite (p.ex. depuis 3s) ne relance AUCUN calcul, le suivi est déjà là et précis.
        floor_t = max(0.0, t0min) if t0min is not None else offset
        if fps and floor_t < seg_t0 - (float(STRIDE) / max(1.0, fps)) - 1e-3:
            s = _seed_snap['s'].copy(); P = _seed_snap['P'].copy()
            tw = _seed_snap['tw']; th = _seed_snap['th']
            tgt_hist = (None if _seed_snap['hist'] is None else _seed_snap['hist'].copy())
            tgt_emb = (None if _seed_snap['emb'] is None else _seed_snap['emb'].copy())
            prev_gray = _seed_snap['gray']; last_cx = _seed_snap['cx']; last_cy = _seed_snap['cy']; coast = 0
            fmin = int(round(floor_t * fps)); fib = seed_fi - STRIDE
            while fib >= fmin:
                if JOBS.get(job_id, {}).get("cancel"): break
                cap.set(cv2.CAP_PROP_POS_FRAMES, fib)
                ret, frame = cap.read()
                if not ret: break
                if USE_PTS:
                    _pm = cap.get(cv2.CAP_PROP_POS_MSEC)
                    tb = (_pm / 1000.0) if (_pm and _pm > 1e-6) else (fib / fps)
                else:
                    tb = fib / fps
                process(frame, tb, fib)
                fib -= STRIDE
            job["msg"] = "Suivi IA (début du clip)…"
        cap.release()
        pts.sort(key=lambda p: p["t"])   # avant + arrière fusionnés -> ordre chronologique
        out = {"fps": fps, "width": W, "height": H, "tracks": [{"id": 0, "n": len(pts), "color": "#00e08f", "points": pts}]}
        (job["dir"] / "tracks.json").write_text(json.dumps(out), encoding="utf-8")
        job["progress"] = 100; job["state"] = "done"; job["msg"] = f"Joueur suivi ✅ ({len(pts)} positions, verrou IA)"
    except Exception as e:
        job["state"] = "error"; job["msg"] = "Erreur : " + str(e); job["trace"] = traceback.format_exc(); print(job["trace"])

@app.post("/track1")
async def track1(vkey: str = Form(...), t0: float = Form(0.0), t1: float = Form(0.0),
                 x: float = Form(0.5), y: float = Form(0.5),
                 model: str = Form("yolo11s.pt"), imgsz: int = Form(640), conf: float = Form(0.25),
                 poly: str = Form(""), cals: str = Form(""), tracker: str = Form("csrt"),
                 bw: float = Form(0.0), bh: float = Form(0.0), t0min: float = Form(-1.0)):
    vp = VKEYS.get(vkey)
    if not vp or not os.path.exists(vp):
        return JSONResponse({"error": "video_absente", "need_file": True}, status_code=409)
    vp = _ensure_seekable(vp); VKEYS[vkey] = vp   # direct WebM -> MP4 seekable (suivi image par image fiable)
    pg = _parse_poly(poly); cl = _parse_cals(cals)
    tk = "nano" if str(tracker).lower().startswith("nano") else "csrt"
    job_id = uuid.uuid4().hex[:10]; d = WORK / job_id; d.mkdir(parents=True, exist_ok=True)
    JOBS[job_id] = {"state": "queued", "progress": 0, "msg": "En file…", "dir": d, "video": vp,
                    "name": vkey, "fps": 0, "w": 0, "h": 0, "frames": 0, "dur": 0, "cancel": False}
    _t0min = None if (t0min is None or float(t0min) < 0) else float(t0min)
    threading.Thread(target=run_track1, args=(job_id, vp, float(t0), float(t1), float(x), float(y),
                     model, int(imgsz), float(conf), pg, cl, tk, float(bw), float(bh), _t0min), daemon=True).start()
    return {"job": job_id}

@app.post("/detect")
def detect(vkey: str = Form(...), t: float = Form(0.0), model: str = Form("yolo11s.pt"),
           imgsz: int = Form(960), conf: float = Form(0.25), poly: str = Form(""), cals: str = Form("")):
    vp = VKEYS.get(vkey)
    if not vp or not os.path.exists(vp):
        return JSONResponse({"error": "video_absente", "need_file": True}, status_code=409)
    try:
        import cv2
        cl = _parse_cals(cals)
        pg = _poly_at(cl, t) if cl else _parse_poly(poly)
        cap = cv2.VideoCapture(vp); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps))); ret, im = cap.read(); cap.release()
        if not ret: return {"boxes": []}
        _pm = get_player_model(); r = _pm.predict(im, conf=conf, classes=getattr(_pm, "_pcls", [0]), verbose=False, **_pred_kw(_pm, imgsz, pick_device()))[0]
        rows = []
        if r.boxes is not None:
            xywh = r.boxes.xywh.cpu().tolist()
            cfs = r.boxes.conf.cpu().tolist() if r.boxes.conf is not None else [1.0] * len(xywh)
            for (cx, cy, w, h), c in zip(xywh, cfs):
                fx = cx / W; fy = min(1.0, (cy + h / 2) / H)
                if pg and not _in_poly(fx, fy, pg):
                    continue  # hors terrain → ignoré
                rows.append((c, {"x": round(fx, 4), "y": round(fy, 4),
                                 "w": round(w / W, 4), "h": round(h / H, 4)}))
        rows.sort(key=lambda r: -r[0])           # plus fiables d'abord
        boxes = [b for _, b in rows[:26]]        # plafond joueurs+arbitre
        return {"boxes": boxes, "w": W, "h": H}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/shots")
def shots(vkey: str = Form(...), t0: float = Form(0.0), t1: float = Form(0.0)):
    """Détecte les changements de plan (coupures caméra) dans [t0,t1].
    Compare des images sous-échantillonnées ~4/s ; un pic de différence = nouvelle prise."""
    vp = VKEYS.get(vkey)
    if not vp or not os.path.exists(vp):
        return JSONResponse({"error": "video_absente", "need_file": True}, status_code=409)
    try:
        import cv2, numpy as np
        cap = cv2.VideoCapture(vp); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = (total / fps) if fps else 0
        a = max(0.0, t0); b = t1 if (t1 and t1 > 0) else dur
        if b <= a: b = dur
        step = 0.25  # 4 images/s
        prev = None; cuts = []; tt = a
        while tt < b:
            cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000.0)
            ok, im = cap.read()
            if not ok: break
            g = cv2.cvtColor(cv2.resize(im, (64, 36)), cv2.COLOR_BGR2GRAY).astype("float32")
            if prev is not None:
                diff = float(np.mean(np.abs(g - prev))) / 255.0
                if diff > 0.18:  # seuil de coupure
                    cuts.append(round(tt, 2))
            prev = g; tt += step
        cap.release()
        # fusionne les pics trop proches (<0.6s)
        merged = []
        for c in cuts:
            if not merged or c - merged[-1] > 0.6:
                merged.append(c)
        return {"cuts": merged, "duration": round(dur, 2)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

def _mp4_ok(path):
    """True si le MP4 s'ouvre et se décode vraiment (rejette les fichiers .mp4 PARTIELS d'un
    encodage interrompu -> 'moov atom not found')."""
    try:
        if not (path and os.path.exists(path) and os.path.getsize(path) > 1024):
            return False
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ff = "ffmpeg"
        import subprocess
        r = subprocess.run([ff, "-v", "error", "-i", path, "-t", "0.1", "-f", "null", "-"],
                           capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False

def _src_dur(path):
    """Durée (s) d'un fichier vidéo, lue via ffmpeg (pour diagnostiquer un fichier optimisé trop court)."""
    try:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ff = "ffmpeg"
        import subprocess, re as _re
        r = subprocess.run([ff, "-i", path], capture_output=True, timeout=60)
        s = (r.stderr or b"").decode("utf-8", "ignore")
        m = _re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", s)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        pass
    return 0.0

def _ensure_seekable(vpath):
    """Les enregistrements du DIRECT sont en WebM/VP9 (MediaRecorder) : le positionnement image par
    image (cap.set POS_MSEC) y est peu fiable (« Not all references are available ») -> le calibrage
    et le suivi extraient des images cassées. On ré-encode donc ces fichiers une fois en MP4 H.264
    avec des images-clés RÉGULIÈRES (seek fiable). Les MP4/MOV normaux sont laissés tels quels."""
    try:
        low = (vpath or "").lower()
        # WebM/MKV (direct) OU tout fichier à FRÉQUENCE VARIABLE (enregistrement d'écran .mov/.mp4) : on
        # normalise UNE fois en MP4 fréquence CONSTANTE (fichier ENTIER -> temps alignés avec la lecture).
        # Les MP4/MOV déjà à fréquence constante (match TV) sont laissés tels quels.
        if not (low.endswith(".webm") or low.endswith(".mkv") or _is_vfr(vpath)):
            return vpath
        out = os.path.splitext(vpath)[0] + "_seekc.mp4"
        if os.path.exists(out) and os.path.getsize(out) > 0:
            if _mp4_ok(out):
                return out
            try:
                os.remove(out)  # fichier partiel/corrompu -> on reconstruit
            except Exception:
                pass
        # -r 30 -vsync cfr : IMPOSE un framerate CONSTANT (WebM MediaRecorder = fps variable qu'OpenCV
        # lit mal). Encodeur : videotoolbox (Apple) puis repli libx264.
        part = out + ".part"
        print("Direct WebM -> ré-encodage MP4 seekable :", os.path.basename(vpath), flush=True)
        if _run_h264_seekable(["-y", "-fflags", "+genpts", "-i", vpath], part):
            try:
                os.replace(part, out)
            except Exception:
                pass
            if os.path.exists(out):
                print("  -> OK :", os.path.basename(out), flush=True)
                return out
        try:
            if os.path.exists(part):
                os.remove(part)
        except Exception:
            pass
        print("  -> échec ré-encodage (aucun encodeur H.264 dispo)", flush=True)
    except Exception as e:
        print("ensure_seekable err:", e, flush=True)
    return vpath

def _prepare_seekable_progress(vpath, pid=""):
    """Prépare UNE fois une copie MP4 propre et seekable du match (comme un fichier téléchargé),
    AVEC progression envoyée à l'app (%). Ré-encode via videotoolbox -> libx264 -> mpeg4, -f mp4
    imposé (sortie .part). Met en cache _seekc.mp4 : les fois suivantes = instantané."""
    low = (vpath or "").lower()
    if not (low.endswith(".webm") or low.endswith(".mkv")):
        return vpath                                  # déjà seekable (téléchargé) -> rien à faire
    out = os.path.splitext(vpath)[0] + "_seekc.mp4"
    if os.path.exists(out) and os.path.getsize(out) > 0 and _mp4_ok(out):
        return out
    import subprocess
    dur = _src_dur(vpath) or 0.0
    part = out + ".part"
    common = ["-pix_fmt", "yuv420p", "-r", "30", "-vsync", "cfr", "-an", "-movflags", "+faststart", "-f", "mp4"]
    variants = [
        ["-c:v", "h264_videotoolbox", "-b:v", "9M", "-g", "30"],
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0"],
        ["-c:v", "mpeg4", "-q:v", "3", "-g", "30"],
    ]
    for ff in _ffmpeg_bins():
        for v in variants:
            try:
                if os.path.exists(part): os.remove(part)
            except Exception:
                pass
            cmd = [ff, "-y", "-fflags", "+genpts", "-i", vpath] + v + common + ["-progress", "pipe:1", "-nostats", part]
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            except Exception:
                continue
            try:
                for line in p.stdout:
                    if pid and dur > 0 and line.startswith("out_time_us="):
                        try:
                            us = int(line.strip().split("=")[1]); pct = min(0.99, us / 1e6 / dur)
                            _cp(pid, 1 + int(55 * pct), "préparation de la vidéo (1re fois)… %d%%" % int(pct * 100))
                        except Exception:
                            pass
                p.wait()
            except Exception:
                try: p.kill()
                except Exception: pass
            if os.path.exists(part) and os.path.getsize(part) > 0 and _mp4_ok(part):
                try: os.replace(part, out)
                except Exception: pass
                if os.path.exists(out):
                    print("  -> copie seekable prête :", os.path.basename(ff), v[1], flush=True)
                    return out
    return _ensure_seekable(vpath)

_VFR_CACHE = {}
def _is_vfr(path):
    """Détecte une vidéo à fréquence VARIABLE (enregistrements d'écran macOS, WebM…). On compare la
    fréquence moyenne et la fréquence de base : si elles diffèrent nettement -> VFR. Le calibrage doit
    alors travailler sur une copie à fréquence CONSTANTE, sinon le positionnement image par image (donc le
    suivi caméra du terrain) est imprécis. Mémorisé par fichier."""
    try:
        key = (path, int(os.path.getmtime(path)))
    except Exception:
        key = (path, 0)
    if key in _VFR_CACHE:
        return _VFR_CACHE[key]
    res = False
    import subprocess
    probes = []
    for ff in _ffmpeg_bins():
        if ff.endswith("ffmpeg"):
            probes.append(ff[:-6] + "ffprobe")
    probes.append(str(BASE / "miniforge" / "bin" / "ffprobe")); probes.append("ffprobe")
    def _f(s):
        try:
            s = s.strip()
            if "/" in s:
                n, d = s.split("/"); d = float(d); return (float(n) / d) if d else 0.0
            return float(s)
        except Exception:
            return 0.0
    for probe in probes:
        try:
            out = subprocess.run([probe, "-v", "error", "-select_streams", "v:0",
                                  "-show_entries", "stream=avg_frame_rate,r_frame_rate",
                                  "-of", "default=nw=1:nk=1", path],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=20).stdout.split()
            if len(out) >= 2:
                avg = _f(out[0]); r = _f(out[1])
                if avg > 0 and r > 0:
                    res = (abs(avg - r) / max(avg, r) > 0.05)
                    break
        except Exception:
            continue
    _VFR_CACHE[key] = res
    try:
        print("  _is_vfr(%s) = %s" % (os.path.basename(path), res), flush=True)
    except Exception:
        pass
    return res

def _seekable_for_window(vpath, a, b, margin=3.0):
    """Prépare une copie MP4 image-par-image UNIQUEMENT pour la FENÊTRE [a-margin, b+margin] du clip.
    Évite de ré-encoder un WebM de 47 min entier (plusieurs minutes) juste pour calibrer/recaler un
    clip de quelques secondes. Retourne (chemin, t_offset) : t_offset = début (s) de la fenêtre dans
    la vidéo d'origine (les temps LOCAUX du fichier = temps d'origine - t_offset)."""
    try:
        low = (vpath or "").lower()
        if not (low.endswith(".webm") or low.endswith(".mkv")):
            if _is_vfr(vpath):
                # VFR .mov/.mp4 (enregistrement d'écran) : copie CFR du FICHIER ENTIER (offset 0, temps
                # alignés avec la lecture) -> le calibrage ne décale plus. Pas de fenêtre -ss (qui décalait).
                return _ensure_seekable(vpath), 0.0
            return vpath, 0.0                       # MP4/MOV déjà à fréquence constante -> temps absolus
        seekc = os.path.splitext(vpath)[0] + "_seekc.mp4"
        if os.path.exists(seekc) and os.path.getsize(seekc) > 0 and _mp4_ok(seekc):
            return seekc, 0.0                       # version complète déjà prête -> temps absolus
        a0 = max(0.0, float(a) - margin)
        dur = ((float(b) - float(a)) if (b and b > a) else 60.0) + 2 * margin
        out = os.path.splitext(vpath)[0] + ("_win_%d_%d.mp4" % (int(round(a0)), int(round(dur))))
        if os.path.exists(out) and os.path.getsize(out) > 0 and _mp4_ok(out):
            return out, a0
        part = out + ".part"
        print("Calibrage : fenêtre seekable %.1f→%.1fs (au lieu de tout le match)…" % (a0, a0 + dur), flush=True)
        if _run_h264_seekable(["-y", "-ss", "%.3f" % a0, "-i", vpath, "-t", "%.3f" % dur], part):
            try: os.replace(part, out)
            except Exception: pass
            if os.path.exists(out):
                return out, a0
        try:
            if os.path.exists(part): os.remove(part)
        except Exception: pass
        print("  -> échec fenêtre, repli sur ré-encodage complet", flush=True)
    except Exception as e:
        print("seekable_window err:", e, flush=True)
    return _ensure_seekable(vpath), 0.0

def _ensure_intra(vpath):
    """Version TOUT-EN-IMAGES-CLÉS (all-intra, comme le Motion JPEG de LiveTag) : chaque image est
    autonome -> le positionnement image par image est QUASI INSTANTANÉ (aucune image à re-décoder
    depuis une image-clé lointaine). Utilisée UNIQUEMENT pour l'export rapide du montage."""
    try:
        low = (vpath or "").lower()
        out = os.path.splitext(vpath)[0] + "_seekI.mp4"
        if out.lower() == low:
            return vpath
        if os.path.exists(out) and os.path.getsize(out) > 0:
            if _mp4_ok(out):
                return out
            try:
                os.remove(out)  # partiel/corrompu -> reconstruction
            except Exception:
                pass
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ff = "ffmpeg"
        import subprocess
        part = out + ".part"
        cmd = [ff, "-y", "-fflags", "+genpts", "-i", vpath,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-r", "30", "-vsync", "cfr",
               "-g", "1", "-keyint_min", "1", "-x264-params", "scenecut=0:keyint=1:min-keyint=1",
               "-an", "-movflags", "+faststart", part]
        print("Export : création MP4 all-intra (seek instantané) :", os.path.basename(vpath), flush=True)
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0 and _mp4_ok(part):
            try:
                os.replace(part, out)
            except Exception:
                pass
            if os.path.exists(out):
                print("  -> OK all-intra :", os.path.basename(out), flush=True)
                return out
        try:
            if os.path.exists(part):
                os.remove(part)
        except Exception:
            pass
        print("  -> échec all-intra (on garde la source) :", (r.stderr[-300:] if r.stderr else b""), flush=True)
    except Exception as e:
        print("ensure_intra err:", e, flush=True)
    return vpath

import threading
FAST_INTRA = {}   # vkey -> chemin MP4 all-intra prêt, ou 'building' pendant l'encodage

def _is_real_video(c):
    """Vrai fichier vidéo utilisable — EXCLUT les temporaires/partiels (.part) et les fenêtres de
    calibrage (_win_*) qui, sinon, peuvent être choisis à la place du vrai match."""
    b = os.path.basename(c).lower()
    if b.endswith(".part") or "_win_" in b:
        return False
    return b.endswith((".webm", ".mkv", ".mp4", ".mov"))

def _resolve_vid(vkey):
    p = VKEYS.get(vkey)
    if p and os.path.exists(p) and _is_real_video(p):
        return p
    try:
        import glob
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", vkey or "")
        if safe:
            cands = [c for c in glob.glob(os.path.join(str(VID), glob.escape(safe) + "*"))
                     if os.path.isfile(c) and _is_real_video(c)]
            seekc = [c for c in cands if c.endswith("_seekc.mp4") and _mp4_ok(c)]
            seek = [c for c in cands if c.endswith("_seek.mp4")]
            # "original" = ni _seekc, ni _seekI, ni _seek. (donc le WebM/MP4 source complet)
            orig = [c for c in cands if not any(s in os.path.basename(c) for s in ("_seekc", "_seekI", "_seek."))]
            pick = seekc or orig or seek or cands
            if pick:
                VKEYS[vkey] = pick[0]
                return pick[0]
    except Exception:
        pass
    return None

def _fast_source_for(p):
    """MP4 seekable DÉJÀ prêt (sans ré-encoder maintenant) pour la source p, sinon None."""
    low = (p or "").lower()
    if low.endswith(".mp4") or low.endswith(".mov"):
        return p
    if low.endswith(".webm") or low.endswith(".mkv"):
        seekc = os.path.splitext(p)[0] + "_seekc.mp4"
        if os.path.exists(seekc) and os.path.getsize(seekc) > 0 and _mp4_ok(seekc):
            return seekc
    return None

def _intra_existing(p):
    base = os.path.splitext(p or "")[0]
    for cand in (base + "_seekI.mp4", base + "_seekc_seekI.mp4"):
        if os.path.exists(cand) and os.path.getsize(cand) > 0:
            if _mp4_ok(cand):
                return cand
            try:
                os.remove(cand)  # fichier optimisé corrompu -> on l'écarte (sera reconstruit)
            except Exception:
                pass
    return None

def _build_intra_bg(vkey, p):
    """Construit (en arrière-plan) le MP4 all-intra rapide : d'abord _seekc.mp4 si WebM, puis all-intra."""
    try:
        s = p
        low = (p or "").lower()
        if low.endswith(".webm") or low.endswith(".mkv"):
            m = _ensure_seekable(s)
            if m and os.path.exists(m):
                s = m
        out = _ensure_intra(s)
        FAST_INTRA[vkey] = out if (out and os.path.exists(out) and out.lower().endswith(".mp4")) else None
    except Exception:
        FAST_INTRA[vkey] = None

@app.post("/put")
async def put(file: UploadFile = File(...), vkey: str = Form(...)):
    """Dépose la vidéo côté serveur SANS lancer d'analyse (pour /shots, /autocal)."""
    ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", vkey or uuid.uuid4().hex)
    vp = VID / (safe + ext)
    os.makedirs(VID, exist_ok=True)   # recrée le cache vidéo si absent (ex. après un nettoyage disque)
    with open(vp, "wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    VKEYS[vkey] = _ensure_seekable(str(vp))
    return {"ok": True}

# ===== ENREGISTREMENT DIRECT : remux -> vidéo lisible/seekable pour le montage =====
REC = WORK / "_rec"; REC.mkdir(exist_ok=True)

def _ffexe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

def _ffmpeg_bins():
    """Liste des ffmpeg à essayer, du plus COMPLET au moins complet : Homebrew (build complet avec
    libx264 + décodeur VP9), puis PATH, puis celui embarqué par imageio (souvent SANS libx264)."""
    bins = []
    try:
        import shutil
        for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if os.path.exists(p) and p not in bins:
                bins.append(p)
        w = shutil.which("ffmpeg")
        if w and w not in bins:
            bins.append(w)
    except Exception:
        pass
    try:
        import imageio_ffmpeg
        ie = imageio_ffmpeg.get_ffmpeg_exe()
        if ie and ie not in bins:
            bins.append(ie)
    except Exception:
        pass
    return bins or ["ffmpeg"]

def _run_h264_seekable(in_args, part):
    """Encode une copie H.264 SEEKABLE (fps constant 30). Essaie plusieurs ffmpeg (Homebrew complet
    -> embarqué) x plusieurs encodeurs (videotoolbox Apple -> libx264). Beaucoup de builds ffmpeg
    (dont imageio) n'ont PAS libx264 et/ou pas de décodeur VP9 -> on cherche une combinaison qui marche."""
    import subprocess
    # -f mp4 : IMPÉRATIF — le fichier de sortie finit par ".part", ffmpeg ne sait pas deviner le
    # format depuis cette extension ("Unable to find output format… Invalid argument"). On l'impose.
    common = ["-pix_fmt", "yuv420p", "-r", "30", "-vsync", "cfr", "-an", "-movflags", "+faststart", "-f", "mp4"]
    variants = [
        ["-c:v", "h264_videotoolbox", "-b:v", "9M", "-g", "30"],
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0"],
        ["-c:v", "mpeg4", "-q:v", "3", "-g", "30"],   # dernier repli (toujours présent)
    ]
    last = b""
    for ff in _ffmpeg_bins():
        for v in variants:
            try:
                if os.path.exists(part): os.remove(part)
            except Exception:
                pass
            try:
                r = subprocess.run([ff] + in_args + v + common + [part], capture_output=True, timeout=14400)
                last = r.stderr or b""
            except Exception as e:
                last = str(e).encode(); continue
            if os.path.exists(part) and os.path.getsize(part) > 0 and _mp4_ok(part):
                print("  -> encodage OK : %s %s" % (os.path.basename(ff), v[1]), flush=True)
                return True
    try:
        print("  -> encodage IMPOSSIBLE. Dernière erreur ffmpeg :", (last[-400:] if last else b""), flush=True)
    except Exception:
        pass
    return False

def _remux_copy(src, out):
    """Ré-encapsule sans ré-encoder (rapide, sans perte) -> ajoute durée + index (cues)."""
    ff = _ffexe()
    cmd = [ff, "-y", "-fflags", "+genpts", "-i", str(src), "-c", "copy", str(out)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=3600)
    except Exception:
        pass
    return os.path.exists(out) and os.path.getsize(out) > 0

def _copy_to_mp4(src, out):
    """FINALISATION QUASI INSTANTANÉE : remise en conteneur MP4 SANS ré-encodage (quelques secondes) quand
    le direct est déjà en H.264/AAC (nouveau moteur WebCodecs). Échoue proprement (retourne False) si la
    source n'est pas copiable telle quelle en MP4 (ex. VP8/WebM) -> on retombe alors sur le ré-encodage complet."""
    ff = _ffexe()
    cmd = [ff, "-y", "-fflags", "+genpts", "-i", str(src),
           "-c", "copy", "-movflags", "+faststart", "-f", "mp4", str(out)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=1200)
    except Exception:
        pass
    return os.path.exists(out) and os.path.getsize(out) > 0 and _mp4_ok(out)

def _reencode_mp4(src, out):
    """Ré-encode le direct en MP4 H.264 SEEKABLE (durée + index + faststart), framerate CONSTANT 30.
    1) accélération matérielle Apple (videotoolbox) = rapide ; 2) repli logiciel libx264."""
    ff = _ffexe()
    base = [ff, "-y", "-fflags", "+genpts", "-i", str(src)]
    tail = ["-pix_fmt", "yuv420p", "-r", "30", "-vsync", "cfr",
            "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k", str(out)]
    # 1) matériel (Mac Apple Silicon / Intel) — très rapide
    hw = base + ["-c:v", "h264_videotoolbox", "-b:v", "22M", "-maxrate", "26M", "-bufsize", "44M"] + tail
    try:
        subprocess.run(hw, capture_output=True, timeout=14400)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return True
    except Exception:
        pass
    # 2) logiciel (repli)
    try:
        if os.path.exists(out):
            os.remove(out)
    except Exception:
        pass
    sw = base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17"] + tail
    try:
        subprocess.run(sw, capture_output=True, timeout=14400)
    except Exception:
        pass
    return os.path.exists(out) and os.path.getsize(out) > 0

@app.post("/rec_remux")
async def rec_remux(file: UploadFile = File(...), rid: str = Form(...)):
    """Reçoit la vidéo d'un direct et la prépare (seekable) pour le montage."""
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", rid or uuid.uuid4().hex)
    os.makedirs(REC, exist_ok=True)
    src = REC / (safe + ".src.webm"); outw = REC / (safe + ".webm"); outm = REC / (safe + ".mp4")
    try:
        with open(src, "wb") as f:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        return JSONResponse({"ok": False, "error": "upload: " + str(e)}, status_code=500)
    url = None
    # MP4 H.264 FIABLE : ré-encodage (matériel videotoolbox, rapide) -> vidéo toujours lisible. La « copie
    # instantanée » a été retirée du chemin : sur certains fichiers elle produisait une vidéo noire.
    if await asyncio.to_thread(_reencode_mp4, src, outm):
        url = "/rec/" + safe + ".mp4"
    elif await asyncio.to_thread(_remux_copy, src, outw):
        url = "/rec/" + safe + ".webm"
    try:
        os.remove(src)
    except Exception:
        pass
    if not url:
        return JSONResponse({"ok": False, "error": "remux echoue (ffmpeg indisponible ?)"}, status_code=500)
    return {"ok": True, "url": url}

@app.get("/rec/{name}")
def rec_get(name: str):
    """Sert la vidéo préparée (FileResponse gère le Range -> seek fluide)."""
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", name or "")
    p = REC / safe
    if not p.exists() or not safe:
        return JSONResponse({"error": "introuvable"}, status_code=404)
    mt = "video/mp4" if safe.endswith(".mp4") else ("video/quicktime" if safe.endswith(".mov") else "video/webm")
    return FileResponse(str(p), media_type=mt)

# ============ CONVERSION AUTO À L'IMPORT (iPhone HEVC/HDR 10 bits -> H.264 1080p SDR) ============
# Le lecteur du navigateur (Chromium) n'affiche PAS le HEVC 10 bits HDR (vidéos iPhone 4K) : on a le
# son mais l'image reste noire. On transcode UNE fois en H.264 8 bits SDR (lisible partout), on met en
# cache (réutilisé à la réouverture) et on lit directement le fichier par son chemin (aucun envoi réseau).
CONV = WORK / "_convert"
try: CONV.mkdir(exist_ok=True)
except Exception: pass
_HDR_TRC = {"smpte2084", "arib-std-b67"}   # HDR10 (PQ) et HLG

def _probe_vinfo(path):
    """Métadonnées de la 1re piste vidéo (ffprobe) : codec, pix_fmt, transfert couleur, dimensions."""
    import subprocess
    probes = []
    for ff in _ffmpeg_bins():
        if ff.endswith("ffmpeg"): probes.append(ff[:-6] + "ffprobe")
    probes.append(str(BASE / "miniforge" / "bin" / "ffprobe")); probes.append("ffprobe")
    for probe in probes:
        try:
            out = subprocess.run([probe, "-v", "error", "-select_streams", "v:0",
                                  "-show_entries", "stream=codec_name,pix_fmt,color_transfer,width,height",
                                  "-of", "default=noprint_wrappers=1", path],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=60).stdout
            d = {}
            for line in out.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1); d[k.strip()] = v.strip()
            if d.get("codec_name"):
                return d
        except Exception:
            continue
    return {}

def _vinfo_needs_h264(info):
    """True s'il faut transcoder : tout sauf du H.264 8 bits SDR (déjà lisible partout)."""
    if not info:
        return False   # illisible -> on ne touche à rien (flux normal)
    codec = (info.get("codec_name") or "").lower()
    pix   = (info.get("pix_fmt") or "").lower()
    trc   = (info.get("color_transfer") or "").lower()
    is10  = ("10" in pix) or ("12" in pix)
    hdr   = trc in _HDR_TRC
    # Codecs que le navigateur (Chromium) lit nativement en 8 bits SDR : on n'y touche pas.
    # Tout le reste (HEVC/H.265 iPhone, 10 bits, HDR, ProRes, mpeg2…) -> transcodage H.264.
    playable = codec in ("h264", "vp8", "vp9", "av1")
    if playable and not is10 and not hdr:
        return False
    return True

def _conv_out_for(path):
    import hashlib
    try:
        st = os.stat(path); key = "%s|%d|%d" % (os.path.abspath(path), st.st_size, int(st.st_mtime))
    except Exception:
        key = os.path.abspath(path or "")
    h = hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()[:16]
    return CONV / (h + ".mp4")

def _convert_to_h264(vpath, pid="", target_h=1080):
    """Transcode vpath -> H.264 8 bits SDR (<=1080p), tonemap HDR->SDR si besoin, progression -> _cp(pid).
    Essaie videotoolbox (GPU Apple) puis libx264 ; tonemap zscale puis repli simple. Cache réutilisé."""
    out = _conv_out_for(vpath)
    if out.exists() and out.stat().st_size > 0 and _mp4_ok(str(out)):
        _cp(pid, 100, "déjà converti"); return str(out)
    info = _probe_vinfo(vpath)
    hdr = (info.get("color_transfer") or "").lower() in _HDR_TRC
    dur = _src_dur(vpath) or 0.0
    part = str(out) + ".part"
    import subprocess
    scale = "scale=-2:%d" % int(target_h)
    vf_tonemap = (scale + ",zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                  "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
    vf_simple  = scale + ",format=yuv420p"
    filters = ([vf_tonemap] if hdr else []) + [vf_simple]
    encs = [["-c:v", "h264_videotoolbox", "-b:v", "8M"],
            ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]]
    tail = ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", "-f", "mp4"]
    for ff in _ffmpeg_bins():
        for vf in filters:
            # si on n'a PAS tonemappé un HDR, on force des tags SDR BT.709 (sinon affichage délavé)
            force_sdr = [] if (hdr and vf == vf_tonemap) else ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
            for enc in encs:
                try:
                    if os.path.exists(part): os.remove(part)
                except Exception: pass
                cmd = [ff, "-y", "-i", vpath, "-vf", vf] + enc + force_sdr + tail + [part]
                try:
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                except Exception:
                    continue
                try:
                    for line in p.stdout:
                        if pid and dur > 0 and line.startswith("out_time_us="):
                            try:
                                us = int(line.strip().split("=")[1]); pct = min(99, int(100 * (us / 1e6) / dur))
                                _cp(pid, pct, "conversion de la vidéo… %d%%" % pct)
                            except Exception: pass
                    p.wait()
                except Exception:
                    try: p.kill()
                    except Exception: pass
                if os.path.exists(part) and os.path.getsize(part) > 0 and _mp4_ok(part):
                    try: os.replace(part, out)
                    except Exception: pass
                    if out.exists():
                        print("  -> conversion H.264 prête :", os.path.basename(ff), enc[1], flush=True)
                        _cp(pid, 100, "conversion terminée")
                        try: _prune_video_cache()
                        except Exception: pass
                        return str(out)
    try:
        if os.path.exists(part): os.remove(part)
    except Exception: pass
    _cp(pid, 100, "conversion impossible")
    return ""

@app.post("/import_check")
async def import_check(path: str = Form(...)):
    """Dit à l'app si la vidéo doit être convertie pour être AFFICHABLE (image), sans rien ré-encoder."""
    if not path or not os.path.exists(path):
        return {"ok": False, "error": "introuvable"}
    info = await asyncio.to_thread(_probe_vinfo, path)
    return {"ok": True, "needs": _vinfo_needs_h264(info),
            "codec": info.get("codec_name", ""), "pix": info.get("pix_fmt", ""),
            "trc": info.get("color_transfer", ""), "w": info.get("width", ""), "h": info.get("height", "")}

@app.post("/import_convert")
async def import_convert(path: str = Form(...), pid: str = Form("")):
    """Transcode la vidéo en H.264 lisible (cache). Progression via /calib_progress?pid=…"""
    if not path or not os.path.exists(path):
        return JSONResponse({"ok": False, "error": "introuvable"}, status_code=404)
    out = await asyncio.to_thread(_convert_to_h264, path, pid)
    if not out:
        return JSONResponse({"ok": False, "error": "conversion impossible"}, status_code=500)
    return {"ok": True, "url": "/conv/" + os.path.basename(out), "path": out}

@app.get("/conv/{name}")
def conv_get(name: str):
    """Sert la vidéo convertie (FileResponse gère le Range -> seek fluide)."""
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", name or "")
    p = CONV / safe
    if not safe or not p.exists():
        return JSONResponse({"error": "introuvable"}, status_code=404)
    return FileResponse(str(p), media_type="video/mp4")

# ===================== SAUVEGARDE DURABLE (principe LiveTag) =====================
# La base de l'app (IndexedDB) peut être purgée par le navigateur si le disque sature. Pour éviter toute perte,
# l'app envoie régulièrement une copie de ses données ; on l'écrit dans un VRAI FICHIER sur le disque
# (jamais purgé), + des copies datées (comme les workspace_backup_<date>.db de LiveTag). Dossier _backups
# à la RACINE (hors cache 'jobs' -> non effacé par « Nettoyer le stockage »).
def _bk_dir():
    d = BASE / "_backups"
    try: d.mkdir(exist_ok=True)
    except Exception: pass
    return d

@app.post("/backup_save")
async def backup_save(file: UploadFile = File(...)):
    try:
        import time as _t
        BK = _bk_dir()
        data = await file.read()
        if not data or len(data) < 2:
            return JSONResponse({"ok": False, "error": "vide"}, status_code=400)
        # copie « vivante » toujours à jour
        (BK / "workspace_latest.json").write_bytes(data)
        # copie datée : au plus UNE PAR JOUR (la copie 'latest' ci-dessus, elle, reste toujours à jour).
        snaps = sorted(BK.glob("workspace_backup_*.json"))
        make = (not snaps) or ((_t.time() - snaps[-1].stat().st_mtime) > 86400)
        if make:
            ts = _t.strftime("%Y-%m-%d_%H%M%S")
            (BK / ("workspace_backup_" + ts + ".json")).write_bytes(data)
            # RÉTENTION PAR PALIERS (bornée à vie, ~27 copies max) :
            #   • les 7 plus récentes (≈ la dernière semaine, jour par jour)
            #   • 1 par SEMAINE sur ~8 semaines
            #   • 1 par MOIS sur ~12 mois
            # Tout le reste est supprimé -> l'espace ne grossit plus avec le temps.
            snaps = sorted(BK.glob("workspace_backup_*.json"))
            keep = set(snaps[-7:])
            now = _t.time()
            seen_week = {}; seen_month = {}
            for p in reversed(snaps[:-7]):
                try:
                    mt = p.stat().st_mtime; lt = _t.localtime(mt)
                    age_days = (now - mt) / 86400.0
                    if age_days <= 60:
                        key = _t.strftime("%Y-W%W", lt)
                        if key not in seen_week and len(seen_week) < 8:
                            seen_week[key] = 1; keep.add(p)
                    else:
                        key = _t.strftime("%Y-%m", lt)
                        if key not in seen_month and len(seen_month) < 12:
                            seen_month[key] = 1; keep.add(p)
                except Exception:
                    keep.add(p)
            for old in snaps:
                if old not in keep:
                    try: old.unlink()
                    except Exception: pass
        return {"ok": True, "bytes": len(data), "dated": bool(make)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/backup_latest")
def backup_latest():
    p = _bk_dir() / "workspace_latest.json"
    if not p.exists():
        return JSONResponse({"ok": False, "error": "aucune sauvegarde"}, status_code=404)
    return FileResponse(str(p), media_type="application/json", headers={"Cache-Control": "no-store"})

@app.get("/backup_list")
def backup_list():
    try:
        BK = _bk_dir(); out = []
        for p in sorted(BK.glob("workspace_backup_*.json"), reverse=True):
            st = p.stat(); out.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime})
        lat = BK / "workspace_latest.json"
        return {"ok": True, "latest_size": (lat.stat().st_size if lat.exists() else 0), "backups": out}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/backup_get")
def backup_get(name: str):
    safe = re.sub(r"[^a-zA-Z0-9_.\-]", "_", name or "")
    p = _bk_dir() / safe
    if not safe or not p.exists():
        return JSONResponse({"ok": False, "error": "introuvable"}, status_code=404)
    return FileResponse(str(p), media_type="application/json", headers={"Cache-Control": "no-store"})

def _encode_montage_mp4(src, out):
    """Ré-encode le rendu du montage en MP4 propre : timestamps régénérés + faststart
    -> durée TOUJOURS correcte, quelle que soit la longueur (corrige le fichier d'1 s)."""
    ff = _ffexe()
    cmd = [ff, "-y", "-fflags", "+genpts", "-i", str(src),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           "-c:a", "aac", "-b:a", "192k", str(out)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=14400)
    except Exception:
        pass
    return os.path.exists(out) and os.path.getsize(out) > 0

@app.post("/montage_encode")
async def montage_encode(file: UploadFile = File(...), fmt: str = Form("mp4")):
    """Reçoit le rendu WebM du montage (navigateur) et renvoie un fichier à durée correcte."""
    uid_ = uuid.uuid4().hex
    os.makedirs(REC, exist_ok=True)
    src = REC / (uid_ + ".src.webm")
    try:
        with open(src, "wb") as f:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        return JSONResponse({"ok": False, "error": "upload: " + str(e)}, status_code=500)
    fmt = (fmt or "mp4").lower()
    ok = False; out = None
    if fmt == "webm":
        out = REC / (uid_ + ".webm"); ok = await asyncio.to_thread(_remux_copy, src, out)
    else:
        ext = "mov" if fmt == "mov" else "mp4"   # même codec H.264/AAC, conteneur selon l'extension
        out = REC / (uid_ + "." + ext); ok = await asyncio.to_thread(_encode_montage_mp4, src, out)
        if not ok:
            out = REC / (uid_ + ".mp4"); ok = await asyncio.to_thread(_reencode_mp4, src, out)
    try:
        os.remove(src)
    except Exception:
        pass
    if not ok or not out:
        return JSONResponse({"ok": False, "error": "encodage echoue (ffmpeg indisponible ?)"}, status_code=500)
    return {"ok": True, "url": "/rec/" + out.name}

# ===== RENDU CÔTÉ SERVEUR (méthode LiveTag) : images JPEG + audio -> ffmpeg =====
RENDER = WORK / "_render"; RENDER.mkdir(exist_ok=True)

def _sid_dir(sid):
    safe = re.sub(r"[^a-f0-9]", "", sid or "")
    if not safe:
        return None, None
    return safe, RENDER / safe

@app.post("/render_start")
async def render_start():
    try: _prune_video_cache(20.0)   # plafond du cache (façon LiveTag) : nettoie les vieilles copies avant de rendre
    except Exception: pass
    sid = uuid.uuid4().hex
    (RENDER / sid).mkdir(parents=True, exist_ok=True)
    return {"ok": True, "sid": sid}

@app.post("/render_frame")
async def render_frame(sid: str = Form(...), idx: int = Form(...), file: UploadFile = File(...)):
    safe, d = _sid_dir(sid)
    if not d or not d.exists():
        return JSONResponse({"ok": False, "error": "session inconnue"}, status_code=400)
    try:
        with open(d / ("f%06d.jpg" % int(idx)), "wb") as f:
            while True:
                ch = await file.read(1 << 20)
                if not ch:
                    break
                f.write(ch)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True}

@app.post("/render_audio")
async def render_audio(sid: str = Form(...), file: UploadFile = File(...)):
    safe, d = _sid_dir(sid)
    if not d or not d.exists():
        return JSONResponse({"ok": False, "error": "session inconnue"}, status_code=400)
    try:
        with open(d / "audio.wav", "wb") as f:
            while True:
                ch = await file.read(1 << 20)
                if not ch:
                    break
                f.write(ch)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True}

def _assemble_render(d, fps, ext):
    ff = _ffexe()
    frames = str(d / "f%06d.jpg"); audio = d / "audio.wav"; out = d / ("out." + ext)
    have_audio = audio.exists()
    # Encodeur du PLUS RAPIDE (GPU Apple = videotoolbox) au plus compatible (CPU). Repli AUTOMATIQUE
    # si l'encodeur matériel n'existe pas dans ce ffmpeg -> on ne perd jamais le rendu.
    vcodecs = [
        ["-c:v", "h264_videotoolbox", "-b:v", "12M", "-pix_fmt", "yuv420p"],
        ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p"],
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"],
    ]
    err = ""
    for vc in vcodecs:
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass
        cmd = [ff, "-y", "-framerate", str(fps), "-i", frames]
        if have_audio:
            cmd += ["-i", str(audio)]
        cmd += vc + ["-movflags", "+faststart", "-r", str(fps)]
        if have_audio:
            # apad : rallonge l'audio au silence -> -shortest coupe sur la vidéo (plus de fin tronquée).
            cmd += ["-af", "apad", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += [str(out)]
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=36000)
            err = (p.stderr.decode("utf-8", "ignore")[-700:] if p.stderr else "")
        except Exception as e:
            err = str(e)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            globals()["_LAST_ENC"] = vc[1]
            return out, err
    return None, err

def _overlay_seq_prepare(d, nframes):
    """Rend la sequence de surcouches CONTINUE (o000000.png..o{n-1}.png) : les images sans dessin
    recoivent un PNG transparent AUX MEMES DIMENSIONS que les surcouches reelles (sinon image2 refuse
    une sequence de tailles differentes). Lien dur = instantane. Renvoie (nb_dessins_reels, ok)."""
    import cv2, numpy as np
    # dimensions d'une surcouche reelle (elles sont en demi-resolution ; ffmpeg les agrandira via scale2ref)
    oh = ow = None
    for i in range(int(nframes)):
        p = d / ("o%06d.png" % i)
        if p.exists():
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is not None:
                oh, ow = im.shape[0], im.shape[1]
                break
    if oh is None:
        return 0, False
    blank = d / "_blank_ov.png"
    try:
        cv2.imwrite(str(blank), np.zeros((oh, ow, 4), np.uint8))
    except Exception:
        return 0, False
    real = 0
    for i in range(int(nframes)):
        p = d / ("o%06d.png" % i)
        if p.exists():
            real += 1
        else:
            try:
                os.link(str(blank), str(p))
            except Exception:
                try:
                    shutil.copy(str(blank), str(p))
                except Exception:
                    return real, False
    return real, True

def _assemble_render_overlay(d, fps, ext, nframes):
    """Assemblage AVEC composition des surcouches PAR FFMPEG en une seule passe (filtre overlay).
    Bien plus rapide que composer chaque image au CPU."""
    real, ok = _overlay_seq_prepare(d, nframes)
    if not ok or real <= 0:
        return None, "pas de surcouches"
    ff = _ffexe()
    frames = str(d / "f%06d.jpg"); ovseq = str(d / "o%06d.png"); audio = d / "audio.wav"; out = d / ("out." + ext)
    have_audio = audio.exists()
    vcodecs = [
        ["-c:v", "h264_videotoolbox", "-b:v", "12M", "-pix_fmt", "yuv420p"],
        ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p"],
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"],
    ]
    err = ""
    for vc in vcodecs:
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass
        _nc = str(os.cpu_count() or 8)
        cmd = [ff, "-y", "-filter_complex_threads", _nc, "-filter_threads", _nc,
               "-framerate", str(fps), "-i", frames,
               "-framerate", str(fps), "-i", ovseq]
        if have_audio:
            cmd += ["-i", str(audio)]
        # scale2ref : agrandit la surcouche (demi-res) à la taille exacte de l'image de fond, puis overlay.
        _fc = "[1:v][0:v]scale2ref=flags=bilinear[ovr][bg];[bg][ovr]overlay=format=auto[v]"
        if have_audio:
            # On RALLONGE l'audio avec du silence (apad) : ainsi -shortest coupe sur la VIDÉO (longueur voulue)
            # et non sur un audio plus court -> plus de fin de montage tronquée (ex. flèche en différé coupée).
            _fc += ";[2:a]apad[aout]"
        cmd += ["-filter_complex", _fc, "-map", "[v]"]
        if have_audio:
            cmd += ["-map", "[aout]"]
        cmd += vc + ["-threads", _nc, "-movflags", "+faststart", "-r", str(fps)]
        if have_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += [str(out)]
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=36000)
            err = (p.stderr.decode("utf-8", "ignore")[-700:] if p.stderr else "")
        except Exception as e:
            err = str(e)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            globals()["_LAST_ENC"] = vc[1]
            return out, err
    return None, err

def _composite_overlays_cv2(d, nframes):
    """REPLI SÛR : compose les surcouches sur les images au CPU (comme avant) si le chemin ffmpeg échoue.
    Ignore les PNG entierement transparents (images sans dessin) -> aucun surcout inutile."""
    import cv2, numpy as np
    for i in range(int(nframes)):
        p = d / ("o%06d.png" % i)
        if not p.exists():
            continue
        ov = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if ov is None or ov.ndim != 3 or ov.shape[2] != 4:
            continue
        try:
            if int(ov[:, :, 3].max()) == 0:
                continue
        except Exception:
            pass
        base_p = d / ("f%06d.jpg" % i)
        base = cv2.imread(str(base_p))
        if base is None:
            continue
        if ov.shape[0] != base.shape[0] or ov.shape[1] != base.shape[1]:
            ov = cv2.resize(ov, (base.shape[1], base.shape[0]))
        a = ov[:, :, 3:4].astype(np.float32) / 255.0
        fg = ov[:, :, :3].astype(np.float32)
        outimg = (fg * a + base.astype(np.float32) * (1.0 - a)).astype(np.uint8)
        cv2.imwrite(str(base_p), outimg, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

@app.post("/render_finish")
async def render_finish(sid: str = Form(...), fps: int = Form(24), fmt: str = Form("mov"), nframes: int = Form(0)):
    safe, d = _sid_dir(sid)
    if not d or not d.exists():
        return JSONResponse({"ok": False, "error": "session inconnue"}, status_code=400)
    # CONTINUITÉ : combler tout trou 0..nframes-1 (sinon l'assembleur image2 s'arrête à la 1re manquante).
    if nframes and int(nframes) > 0:
        last = None
        holes = 0
        for i in range(int(nframes)):
            fp = d / ("f%06d.jpg" % i)
            if fp.exists():
                last = fp
            elif last is not None:
                try:
                    shutil.copy(str(last), str(fp)); holes += 1
                except Exception:
                    pass
        if holes:
            print("render_finish : %d image(s) comblée(s) sur %d." % (holes, int(nframes)), flush=True)
    ext = "mov" if (fmt or "mov") == "mov" else ("webm" if fmt == "webm" else "mp4")
    if ext == "webm":
        ext = "mp4"  # H.264 -> conteneur mp4 (webm exigerait vp9)
    # Y a-t-il des surcouches STOCKÉES (o*.png) -> mode ffmpeg-overlay (rapide). Sinon assemblage normal.
    ov_files = list(d.glob("o[0-9]*.png"))
    if ov_files:
        out, err = await asyncio.to_thread(_assemble_render_overlay, d, int(fps), ext, int(nframes))
        if not out:
            # REPLI SÛR : on compose les dessins au CPU (aucune perte de qualité), puis assemblage normal.
            try:
                await asyncio.to_thread(_composite_overlays_cv2, d, int(nframes))
            except Exception:
                pass
            out2, err2 = await asyncio.to_thread(_assemble_render, d, int(fps), ext)
            out = out2; err = "overlay KO (" + (err or "") + ") -> repli CPU: " + (err2 or "")
    else:
        out, err = await asyncio.to_thread(_assemble_render, d, int(fps), ext)
    if not out:
        return JSONResponse({"ok": False, "error": "ffmpeg: " + (err or "assemblage echoue")}, status_code=500)
    # DIAG durée : audio.wav et sortie (pour vérifier une éventuelle troncature en fin de montage).
    _auddur = 0.0; _outdur = 0.0
    try:
        import wave as _wave
        _ap = d / "audio.wav"
        if _ap.exists():
            with _wave.open(str(_ap), "rb") as _w:
                _auddur = _w.getnframes() / float(_w.getframerate() or 1)
    except Exception:
        pass
    try:
        _outdur = (int(nframes) / float(fps)) if (nframes and fps) else 0.0
    except Exception:
        pass
    try:
        print("render_finish diag : nframes=%s fps=%s -> video=%.2fs · audio.wav=%.2fs" % (nframes, fps, _outdur, _auddur), flush=True)
    except Exception:
        pass
    dest = REC / (safe + "." + ext); ok_dest = False
    try:
        shutil.move(str(out), str(dest)); ok_dest = True
    except Exception:
        try:
            shutil.copy(str(out), str(dest)); ok_dest = True
        except Exception:
            dest = out; ok_dest = False
    if ok_dest:   # on ne nettoie le dossier de travail QUE si la sortie est bien copiée dans REC (servi par /rec)
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
    return {"ok": True, "url": "/rec/" + dest.name, "sv": SERVER_VERSION, "enc": globals().get("_LAST_ENC", "?")}

# ===================== MOTEUR TURBO : montage par SEGMENTS (pas image par image) =====================
# Idée : au lieu de matérialiser TOUTES les images en JPEG puis de tout ré-encoder (lent), on encode
# chaque « run » en UNE pièce -> un gel = 1 image TENUE (quasi gratuit), un bout de match = coupe DIRECTE
# depuis la source (un seul décodage), puis on COLLE les pièces (concat). Repli auto sur le moteur image
# par image si le moindre souci. Les images des gels viennent du navigateur (identiques à l'ancien rendu).
_TURBO_ENCODERS = [
    # Mode « temps réel » de l'encodeur Apple : plus rapide, image identique à l'œil à 12 Mb/s.
    ["-c:v", "h264_videotoolbox", "-b:v", "12M", "-realtime", "1", "-pix_fmt", "yuv420p"],
    # Repli si -realtime n'est pas reconnu par ce ffmpeg -> videotoolbox normal (= vitesse actuelle, aucun risque).
    ["-c:v", "h264_videotoolbox", "-b:v", "12M", "-pix_fmt", "yuv420p"],
    ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"],
    ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p"],
]

def _turbo_scale_vf(w, h):
    return ("scale=%d:%d:force_original_aspect_ratio=decrease,"
            "pad=%d:%d:(ow-iw)/2:(oh-ih)/2:black,setsar=1") % (w, h, w, h)

def _turbo_piece(ff, run, d, k, vc, vf, fps, vkey):
    """Encode UNE pièce (.mp4 H.264) pour un run. Renvoie (chemin, nframes) ou (None, err)."""
    n = int(round(float(run.get("dur", 0.0)) * fps))
    if n <= 0:
        return None, "run vide"
    out = str(d / ("p_%05d.mp4" % k))
    try:
        if os.path.exists(out):
            os.remove(out)
    except Exception:
        pass
    typ = run.get("type")
    if typ == "still":
        img = d / ("s_%s.jpg" % str(run.get("img", "")))
        if not img.exists():
            return None, "image de gel absente (%s)" % run.get("img")
        cmd = [ff, "-y", "-loop", "1", "-t", ("%.4f" % (n / float(fps) + 0.05)),
               "-i", str(img), "-vf", vf] + vc + \
              ["-r", str(fps), "-frames:v", str(n), "-an", out]
    else:  # match : coupe directe depuis la source
        _vk = (str(run.get("vk") or "")).strip() or vkey
        src = _export_src(_vk)
        if not (src and os.path.exists(src)):
            return None, "source match absente (%s)" % _vk
        ts0 = float(run.get("ts0", 0.0))
        cmd = [ff, "-y", "-ss", ("%.4f" % max(0.0, ts0)), "-i", src,
               "-t", ("%.4f" % (n / float(fps) + 0.20)),
               "-vf", (vf + (",fps=%d" % fps))] + vc + \
              ["-r", str(fps), "-frames:v", str(n), "-vsync", "cfr", "-an", out]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=240)
    except Exception as e:
        return None, str(e)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out, n
    return None, (p.stderr.decode("utf-8", "ignore")[-300:] if p.stderr else "pièce vide")

def _turbo_piece_any(run, d, k, vf, fps, vkey):
    """Encode UNE pièce. On VERROUILLE l'encodeur choisi (1re pièce réussie) pour TOUTE la session -> toutes
    les pièces sont homogènes -> le collage final se fait en COPIE directe (instantané). Repli GPU Apple -> CPU."""
    ff = _ffexe()
    encfile = d / "enc.idx"
    forced = None
    try:
        if encfile.exists():
            forced = int(encfile.read_text().strip())
    except Exception:
        forced = None
    order = list(range(len(_TURBO_ENCODERS)))
    idxs = [forced] if (forced is not None and 0 <= forced < len(_TURBO_ENCODERS)) else order
    err = ""
    for i in idxs:
        vc = _TURBO_ENCODERS[i]
        pth, info = _turbo_piece(ff, run, d, k, vc, vf, fps, vkey)
        if pth:
            globals()["_LAST_ENC"] = vc[1]
            if forced is None:
                try:
                    encfile.write_text(str(i))
                except Exception:
                    pass
            return pth, info
        err = info or err
    # l'encodeur verrouillé a échoué sur cette pièce -> dernier recours : essayer les autres
    if forced is not None:
        for i in order:
            if i == forced:
                continue
            vc = _TURBO_ENCODERS[i]
            pth, info = _turbo_piece(ff, run, d, k, vc, vf, fps, vkey)
            if pth:
                globals()["_LAST_ENC"] = vc[1]
                return pth, info
            err = info or err
    return None, err

def _turbo_concat(d, fps, ext, total_n=0):
    """Colle les pièces p_*.mp4 (dans l'ordre) + audio. concat -c copy puis repli ré-encodé."""
    ff = _ffexe()
    _dur = (float(total_n) / float(fps)) if (total_n and fps) else 0.0
    import glob as _glob
    pieces = sorted(_glob.glob(str(d / "p_*.mp4")))
    if not pieces:
        return None, "aucune pièce"
    audio = d / "audio.wav"
    have_audio = audio.exists()
    lst = d / "concat.txt"
    try:
        with open(lst, "w") as f:
            for p in pieces:
                f.write("file '%s'\n" % os.path.abspath(p).replace("\\", "/"))
    except Exception as e:
        return None, "liste concat: " + str(e)
    out = d / ("turbo_out." + ext)
    for mode in ("copy", "reencode"):
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass
        cmd = [ff, "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
        if have_audio:
            cmd += ["-i", str(audio)]
        if mode == "copy":
            cmd += ["-c:v", "copy"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(fps)]
        if have_audio:
            cmd += ["-af", "apad", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        if _dur > 0:
            cmd += ["-t", ("%.3f" % _dur)]
        cmd += ["-movflags", "+faststart", str(out)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=36000)
        except Exception as e:
            return None, str(e)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out, ""
    return None, "concat échoué"

@app.post("/render_turbo_piece")
async def render_turbo_piece(sid: str = Form(...), k: int = Form(...), run: str = Form(...),
                             vkey: str = Form(""), w: int = Form(...), h: int = Form(...),
                             fps: int = Form(24)):
    """Construit UNE pièce (un run). Appelé en boucle par le navigateur (progression + Stop possibles)."""
    safe, d = _sid_dir(sid)
    if not d or not d.exists():
        return JSONResponse({"ok": False, "error": "session inconnue"}, status_code=400)
    try:
        import json as _json
        r = _json.loads(run or "{}")
    except Exception as e:
        return JSONResponse({"ok": False, "error": "run: " + str(e)}, status_code=400)
    vf = _turbo_scale_vf(int(w), int(h))
    pth, info = await asyncio.to_thread(_turbo_piece_any, r, d, int(k), vf, int(fps), vkey)
    if not pth:
        return JSONResponse({"ok": False, "error": info or "pièce échouée"}, status_code=500)
    return {"ok": True, "enc": globals().get("_LAST_ENC", "?")}

@app.post("/render_turbo_finish")
async def render_turbo_finish(sid: str = Form(...), fps: int = Form(24), fmt: str = Form("mov"), nframes: int = Form(0)):
    """Colle toutes les pièces déjà construites + audio -> vidéo finale."""
    safe, d = _sid_dir(sid)
    if not d or not d.exists():
        return JSONResponse({"ok": False, "error": "session inconnue"}, status_code=400)
    ext = "mov" if (fmt or "mov") == "mov" else "mp4"
    out, err = await asyncio.to_thread(_turbo_concat, d, int(fps), ext, int(nframes))
    if not out:
        return JSONResponse({"ok": False, "error": err or "assemblage échoué"}, status_code=500)
    dest = REC / (safe + "." + ext)
    ok_dest = False
    try:
        shutil.move(str(out), str(dest)); ok_dest = True
    except Exception:
        try:
            shutil.copy(str(out), str(dest)); ok_dest = True
        except Exception:
            dest = out; ok_dest = False
    if ok_dest:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
    return {"ok": True, "url": "/rec/" + dest.name, "sv": SERVER_VERSION,
            "enc": globals().get("_LAST_ENC", "?"), "nframes": int(nframes)}

def _render_turbo_build(d, vkey, w, h, fps, ext, runs):
    """Construit la vidéo par pièces + concat. Renvoie (chemin_sortie, err, nframes_total)."""
    ff = _ffexe()
    vf = _turbo_scale_vf(w, h)
    audio = d / "audio.wav"
    have_audio = audio.exists()
    # Choix de l'encodeur : on tente la 1re pièce avec chaque encodeur jusqu'à ce qu'un marche,
    # puis on garde CE MÊME encodeur pour toutes les pièces (concat -c copy exige des pièces homogènes).
    chosen = None
    err = ""
    for vc in _TURBO_ENCODERS:
        p0, info0 = _turbo_piece(ff, runs[0], d, 0, vc, vf, fps, vkey)
        if p0:
            chosen = vc
            break
        err = info0 or err
    if chosen is None:
        return None, ("aucun encodeur: " + (err or "?")), 0
    globals()["_LAST_ENC"] = chosen[1]
    pieces = [str(d / "p_00000.mp4")]
    total_n = int(round(float(runs[0].get("dur", 0.0)) * fps))
    for k in range(1, len(runs)):
        pth, info = _turbo_piece(ff, runs[k], d, k, chosen, vf, fps, vkey)
        if not pth:
            return None, ("pièce %d: %s" % (k, info)), 0
        pieces.append(pth)
        total_n += int(round(float(runs[k].get("dur", 0.0)) * fps))
    # liste de concat
    lst = d / "concat.txt"
    try:
        with open(lst, "w") as f:
            for p in pieces:
                f.write("file '%s'\n" % os.path.abspath(p).replace("\\", "/"))
    except Exception as e:
        return None, "liste concat: " + str(e), 0
    out = d / ("turbo_out." + ext)
    # 1) concat SANS ré-encodage (rapide) + mux audio
    cmd = [ff, "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if have_audio:
        cmd += ["-i", str(audio)]
    cmd += ["-c:v", "copy"]
    if have_audio:
        cmd += ["-af", "apad", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    if total_n > 0:
        cmd += ["-t", ("%.3f" % (float(total_n) / float(fps)))]
    cmd += ["-movflags", "+faststart", str(out)]
    ok = False
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=36000)
        ok = os.path.exists(out) and os.path.getsize(out) > 0
        if not ok:
            err = (p.stderr.decode("utf-8", "ignore")[-300:] if p.stderr else "")
    except Exception as e:
        err = str(e)
    # 2) repli : concat AVEC ré-encodage (au cas où -c copy refuse selon l'encodeur)
    if not ok:
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass
        cmd = [ff, "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
        if have_audio:
            cmd += ["-i", str(audio)]
        cmd += chosen + ["-r", str(fps)]
        if have_audio:
            cmd += ["-af", "apad", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        if total_n > 0:
            cmd += ["-t", ("%.3f" % (float(total_n) / float(fps)))]
        cmd += ["-movflags", "+faststart", str(out)]
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=36000)
            ok = os.path.exists(out) and os.path.getsize(out) > 0
            if not ok:
                err = (p.stderr.decode("utf-8", "ignore")[-300:] if p.stderr else err)
        except Exception as e:
            err = str(e)
    if not ok:
        return None, ("assemblage turbo: " + (err or "?")), 0
    # nettoyage des pièces (on garde la sortie)
    for p in pieces:
        try:
            os.remove(p)
        except Exception:
            pass
    return out, "", total_n

@app.post("/render_still")
async def render_still(sid: str = Form(...), img: str = Form(...), file: UploadFile = File(...)):
    """Stocke UNE image de gel (réutilisée par plusieurs runs). id = hachage contenu côté navigateur."""
    safe, d = _sid_dir(sid)
    if not d or not d.exists():
        return JSONResponse({"ok": False, "error": "session inconnue"}, status_code=400)
    dst = d / ("s_%s.jpg" % str(img))
    try:
        data = await file.read()
        with open(dst, "wb") as f:
            f.write(data)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True}

@app.post("/register_local")
def register_local(vkey: str = Form(...), path: str = Form(...)):
    """MÉTHODE LIVETAG : au lieu de COPIER la vidéo dans le cache, on enregistre le CHEMIN du fichier
    d'origine (sur le Mac ou un disque externe). Le moteur lit alors directement l'original -> AUCUNE copie,
    aucun espace disque pris en double. Si le fichier a été déplacé/supprimé, on répond need_file."""
    try:
        p = (path or "").strip()
        if not p or not os.path.exists(p):
            return JSONResponse({"ok": False, "error": "fichier introuvable", "need_file": True}, status_code=404)
        if not _is_real_video(p):
            return JSONResponse({"ok": False, "error": "fichier vidéo illisible"}, status_code=400)
        VKEYS[vkey] = p   # lecture directe de l'original (référence, pas de copie)
        try:
            print("Référence directe (LiveTag) :", vkey, "->", p, flush=True)
        except Exception:
            pass
        return {"ok": True, "path": p}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/render_turbo")
async def render_turbo(sid: str = Form(...), vkey: str = Form(""),
                       w: int = Form(...), h: int = Form(...),
                       fps: int = Form(24), fmt: str = Form("mov"),
                       runs: str = Form(...)):
    """Assemblage TURBO par pièces (gel = image tenue, match = coupe directe) puis concat."""
    safe, d = _sid_dir(sid)
    if not d or not d.exists():
        return JSONResponse({"ok": False, "error": "session inconnue"}, status_code=400)
    try:
        import json as _json
        runlist = _json.loads(runs or "[]")
    except Exception as e:
        return JSONResponse({"ok": False, "error": "runs: " + str(e)}, status_code=400)
    if not runlist:
        return JSONResponse({"ok": False, "error": "aucun run"}, status_code=400)
    ext = "mov" if (fmt or "mov") == "mov" else ("mp4" if fmt in ("webm", "mp4") else "mp4")
    out, err, total_n = await asyncio.to_thread(_render_turbo_build, d, vkey, int(w), int(h), int(fps), ext, runlist)
    if not out:
        return JSONResponse({"ok": False, "error": err or "turbo échoué"}, status_code=500)
    dest = REC / (safe + "." + ext)
    ok_dest = False
    try:
        shutil.move(str(out), str(dest)); ok_dest = True
    except Exception:
        try:
            shutil.copy(str(out), str(dest)); ok_dest = True
        except Exception:
            dest = out; ok_dest = False
    if ok_dest:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
    return {"ok": True, "url": "/rec/" + dest.name, "sv": SERVER_VERSION,
            "enc": globals().get("_LAST_ENC", "?"), "nframes": total_n}

# ===================== MOTEUR RAPIDE (façon LiveTag) : décodage NATIF côté serveur =====================
def _ensure_timeline_mp4(vpath):
    """MP4 H.264 PROPRE qui PRÉSERVE la timeline d'origine (les pauses du direct deviennent des images
    tenues) -> à la fois seekable de façon fiable ET aligné en temps avec le montage. Construit UNE fois.
    On N'UTILISE PAS +genpts (qui SUPPRIME les pauses et raccourcit la vidéo, ex. 2843s -> 1790s) ; le
    filtre fps tient la dernière image pendant les trous, ce qui conserve la durée réelle."""
    try:
        low = (vpath or "").lower()
        if not (low.endswith(".webm") or low.endswith(".mkv")):
            return vpath  # déjà un MP4/MOV lisible
        out = os.path.splitext(vpath)[0] + "_tl.mp4"
        if os.path.exists(out) and os.path.getsize(out) > 0 and _mp4_ok(out):
            return out
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ff = "ffmpeg"
        import subprocess
        part = out + ".part"
        cmd = [ff, "-y", "-err_detect", "ignore_err", "-i", vpath,
               "-vf", "fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-g", "30", "-keyint_min", "30",
               "-an", "-movflags", "+faststart", part]
        print("Export : création MP4 timeline-préservée (seekable) :", os.path.basename(vpath), flush=True)
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0 and _mp4_ok(part):
            try:
                os.replace(part, out)
            except Exception:
                pass
            if os.path.exists(out):
                print("  -> OK timeline :", os.path.basename(out), flush=True)
                return out
        try:
            if os.path.exists(part):
                os.remove(part)
        except Exception:
            pass
        print("  -> échec timeline (on garde l'original) :", (r.stderr[-300:] if r.stderr else b""), flush=True)
    except Exception as e:
        print("ensure_timeline err:", e, flush=True)
    return vpath

def _original_vid(vkey):
    """Fichier vidéo D'ORIGINE (webm/mkv/mp4/mov), PAS un dérivé (_seekc / _seekI). C'est lui qui a la
    MÊME timeline que le montage (le ré-encodage CFR supprime les pauses du direct -> durée différente)."""
    p = _resolve_vid(vkey)
    if not p:
        return None
    b = os.path.basename(p)
    if "_seekc" in b or "_seekI" in b:
        base = os.path.splitext(p)[0]
        for suf in ("_seekc_seekI", "_seekI", "_seekc"):
            if base.endswith(suf):
                base = base[:-len(suf)]
                break
        for ext in (".webm", ".mkv", ".mp4", ".mov"):
            cand = base + ext
            if os.path.exists(cand):
                return cand
    return p

def _export_src(vkey):
    """Source pour l'extraction native. On prend EXACTEMENT le fichier que le navigateur joue via /vid
    (all-intra _seekI.mp4 > _seekc.mp4 > source) : même timeline que le montage ET seekable rapidement
    partout (H.264, images-clés régulières, faststart). Rien à ré-encoder au moment de l'export."""
    p = _resolve_vid(vkey)
    if not (p and os.path.exists(p)):
        return None
    fast = FAST_INTRA.get(vkey)
    if isinstance(fast, str) and os.path.exists(fast):
        return fast
    cand = _intra_existing(p)
    if cand:
        return cand
    s = _fast_source_for(p)
    if s and os.path.exists(s):
        return s
    return p

@app.post("/cut_clip")
def cut_clip(vkey: str = Form(...), t0: float = Form(...), t1: float = Form(...),
             name: str = Form("clip"), fmt: str = Form("mp4"), voice: UploadFile = File(None)):
    """Découpe le segment [t0,t1] de la vidéo source -> MP4 téléchargeable.
    - SANS voix jointe : on garde le SON ORIGINAL de la vidéo.
    - AVEC voix jointe (clip commenté) : on COUPE le son original et on ne garde QUE la voix,
      calée au début du clip (comme à la lecture). Ré-encodage H.264/AAC, découpage précis à l'image."""
    import tempfile, time as _time
    src = _resolve_vid(vkey)
    if not (src and os.path.exists(src)):
        src = _export_src(vkey)
    if not (src and os.path.exists(src)):
        return JSONResponse({"ok": False, "error": "source video absente (importe-la d'abord)"}, status_code=400)
    ff = _ffexe()
    dur = max(0.1, float(t1) - float(t0))
    safe = (re.sub(r'[^A-Za-z0-9_.-]+', '_', (name or 'clip')).strip('_') or 'clip')[:60]
    _ext = "mov" if str(fmt).lower() == "mov" else "mp4"
    out = os.path.join(tempfile.gettempdir(), "matchlyz_cut_%d.%s" % (int(_time.time() * 1000), _ext))
    # Voix éventuelle -> fichier temporaire
    vpath = None
    if voice is not None:
        try:
            data = voice.file.read()
            if data:
                vpath = os.path.join(tempfile.gettempdir(), "matchlyz_voice_%d.webm" % int(_time.time() * 1000))
                with open(vpath, "wb") as _vf:
                    _vf.write(data)
        except Exception:
            vpath = None
    if vpath:
        # Vidéo (sans son original) + voix (à partir de 0 = début du clip), bornée à la durée du clip.
        cmd = [ff, "-y", "-ss", ("%.4f" % max(0.0, float(t0))), "-i", src, "-i", vpath,
               "-t", ("%.4f" % dur), "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out]
    else:
        cmd = [ff, "-y", "-ss", ("%.4f" % max(0.0, float(t0))), "-i", src,
               "-t", ("%.4f" % dur),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out]
    try:
        pr = subprocess.run(cmd, capture_output=True, timeout=36000)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        if vpath:
            try: os.remove(vpath)
            except Exception: pass
    if not (os.path.exists(out) and os.path.getsize(out) > 0):
        err = pr.stderr.decode("utf-8", "ignore")[-400:] if (pr and pr.stderr) else "echec ffmpeg"
        return JSONResponse({"ok": False, "error": err}, status_code=500)
    return FileResponse(out, media_type=("video/quicktime" if _ext == "mov" else "video/mp4"),
                        filename=safe + "." + _ext, headers={"Cache-Control": "no-store"})

@app.post("/render_match")
def render_match(sid: str = Form(...), vkey: str = Form(...),
                 w: int = Form(...), h: int = Form(...), segs: str = Form(...)):
    """Extrait NATIVEMENT (ffmpeg) les images du match pour des SEGMENTS contigus, mises à l'échelle
    'contain' (letterbox) en w×h -> images de base f%06d.jpg. Aucune image seekée côté navigateur."""
    safe, d = _sid_dir(sid)
    if not d or not d.exists():
        return JSONResponse({"ok": False, "error": "session inconnue"}, status_code=400)
    try:
        import json as _json
        segments = _json.loads(segs or "[]")
    except Exception as e:
        return JSONResponse({"ok": False, "error": "segments: " + str(e)}, status_code=400)
    ff = _ffexe()
    vf_scale = ("scale=%d:%d:force_original_aspect_ratio=decrease,"
                "pad=%d:%d:(ow-iw)/2:(oh-ih)/2:black") % (w, h, w, h)
    made = 0
    want = 0
    shorts = []
    last_err = ""
    for seg in segments:
        try:
            i0 = int(seg["i0"]); count = int(seg["count"])
            want += count
            # Source PAR SEGMENT : "vk" fourni (vidéo importée) sinon la vidéo du match (vkey global).
            _vk = (str(seg.get("vk") or "")).strip() or vkey
            src = _export_src(_vk)
            if not (src and os.path.exists(src)):
                last_err = "source video absente (" + str(_vk) + ")"; continue
            ts0 = float(seg["ts0"]); dts = float(seg["dts"]) or (1.0 / 24.0)
            rate = 1.0 / dts if dts > 0 else 24.0
            tmp = str(d / ("m_%d_%%06d.jpg" % i0))
            vf = ("fps=%.6f," % rate) + vf_scale
            # Fenêtre bornée par le TEMPS (pas par un nombre d'images) : plus robuste sur les zones à
            # timing irrégulier des WebM du direct. -ss AVANT -i (ffmpeg 6) = rapide ET précis.
            seg_dur = (count / rate) + 0.20 if rate > 0 else 1.0
            cmd = [ff, "-y", "-fflags", "+genpts", "-ss", ("%.4f" % max(0.0, ts0)), "-i", src,
                   "-t", ("%.4f" % seg_dur), "-vf", vf, "-q:v", "2", tmp]
            pr = subprocess.run(cmd, capture_output=True, timeout=36000)
            if pr.returncode != 0 and pr.stderr:
                last_err = pr.stderr.decode("utf-8", "ignore")[-500:]
            got = 0
            for k in range(count):
                srcf = d / ("m_%d_%06d.jpg" % (i0, k + 1))
                if srcf.exists():
                    try:
                        os.replace(str(srcf), str(d / ("f%06d.jpg" % (i0 + k))))
                        made += 1; got += 1
                    except Exception:
                        pass
            # REPLI (zones à références cassées du WebM du direct : ffmpeg cale au seek direct) :
            # on RÉ-ENCODE juste ce petit bout en MP4 propre (décodage LINÉAIRE tolérant, quelques s),
            # puis on en extrait les images AVEC le mouvement. Bien plus rapide que refaire au navigateur.
            if got == 0:
                PAD = 2.0
                clip = str(d / ("clip_%d.mp4" % i0))
                cs = max(0.0, ts0 - PAD); off = ts0 - cs
                cc = [ff, "-y", "-fflags", "+genpts", "-err_detect", "ignore_err",
                      "-ss", ("%.4f" % cs), "-i", src, "-t", ("%.4f" % (off + seg_dur + 0.4)),
                      "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                      "-pix_fmt", "yuv420p", "-vsync", "cfr", "-r", "30", clip]
                try:
                    subprocess.run(cc, capture_output=True, timeout=36000)
                except Exception:
                    pass
                if os.path.exists(clip) and os.path.getsize(clip) > 0:
                    c2 = [ff, "-y", "-ss", ("%.4f" % off), "-i", clip, "-t", ("%.4f" % seg_dur),
                          "-vf", vf, "-q:v", "2", tmp]
                    try:
                        subprocess.run(c2, capture_output=True, timeout=36000)
                    except Exception:
                        pass
                    for k in range(count):
                        srcf = d / ("m_%d_%06d.jpg" % (i0, k + 1))
                        if srcf.exists():
                            try:
                                os.replace(str(srcf), str(d / ("f%06d.jpg" % (i0 + k)))); made += 1; got += 1
                            except Exception:
                                pass
                    try:
                        os.remove(clip)
                    except Exception:
                        pass
                # dernier recours : une image maintenue (au moins la bonne, pas une d'avant)
                if got == 0:
                    one = str(d / ("one_%d.jpg" % i0))
                    try:
                        subprocess.run([ff, "-y", "-fflags", "+genpts", "-err_detect", "ignore_err",
                                        "-ss", ("%.4f" % max(0.0, ts0)), "-i", src, "-frames:v", "1",
                                        "-vf", vf_scale, "-q:v", "2", one], capture_output=True, timeout=36000)
                    except Exception:
                        pass
                    if os.path.exists(one):
                        for k in range(count):
                            try:
                                shutil.copy(one, str(d / ("f%06d.jpg" % (i0 + k)))); made += 1; got += 1
                            except Exception:
                                pass
                        try:
                            os.remove(one)
                        except Exception:
                            pass
            # CONTINUITÉ : combler tout trou en dupliquant l'image PRÉCÉDENTE (globale, y compris i0-1)
            for k in range(count):
                fp = d / ("f%06d.jpg" % (i0 + k))
                if not fp.exists():
                    prev = d / ("f%06d.jpg" % (i0 + k - 1))
                    if prev.exists():
                        try:
                            shutil.copy(str(prev), str(fp))
                        except Exception:
                            pass
            if got < count and len(shorts) < 12:
                shorts.append({"i0": i0, "count": count, "got": got, "ts0": round(ts0, 2)})
        except Exception as e:
            last_err = str(e)
            continue
    if made == 0:
        return JSONResponse({"ok": False, "made": 0, "want": want,
                             "error": "extraction native vide — source=" + os.path.basename(src or "?") +
                             " · " + (last_err or "(pas de détail ffmpeg)")}, status_code=500)
    return {"ok": True, "made": made, "want": want, "shorts": shorts,
            "src": os.path.basename(src or ""), "dur": round(_src_dur(src), 1),
            "err": (last_err[-200:] if last_err else "")}

@app.post("/render_overlay")
def render_overlay(sid: str = Form(...), idx: int = Form(...), file: UploadFile = File(...)):
    """Compose une surcouche transparente (dessins/télestration, déjà en coordonnées de sortie)
    par-dessus l'image de base f{idx}.jpg produite par /render_match ou /render_frame.
    SYNCHRONE volontairement : FastAPI l'exécute dans son pool de threads -> plusieurs surcouches
    s'appliquent EN PARALLÈLE (OpenCV libère le GIL), au lieu de se bloquer sur la boucle asyncio."""
    safe, d = _sid_dir(sid)
    if not d or not d.exists():
        return JSONResponse({"ok": False, "error": "session inconnue"}, status_code=400)
    try:
        import numpy as np, cv2
        data = file.file.read()
        ov = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if ov is None:
            return {"ok": False, "error": "overlay illisible"}
        base_p = d / ("f%06d.jpg" % int(idx))
        base = cv2.imread(str(base_p)) if base_p.exists() else None
        if base is None:
            H2 = ov.shape[0]; W2 = ov.shape[1]
            base = np.zeros((H2, W2, 3), np.uint8)
        if ov.shape[0] != base.shape[0] or ov.shape[1] != base.shape[1]:
            ov = cv2.resize(ov, (base.shape[1], base.shape[0]))
        if ov.ndim == 3 and ov.shape[2] == 4:
            a = ov[:, :, 3:4].astype(np.float32) / 255.0
            fg = ov[:, :, :3].astype(np.float32)
            out = fg * a + base.astype(np.float32) * (1.0 - a)
            out = out.astype(np.uint8)
        else:
            out = base
        cv2.imwrite(str(base_p), out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True}

@app.post("/render_overlay_multi")
def render_overlay_multi(sid: str = Form(...), idxs: str = Form(...), file: UploadFile = File(...)):
    """Applique UNE surcouche (dessin) transparente à PLUSIEURS images de base d'un coup.
    Sert à la déduplication : un dessin statique (flèche/zone/texte fixe) est identique sur
    des dizaines/centaines d'images -> on ne l'encode/envoie qu'UNE fois puis on le composite
    sur toute la plage côté serveur. Bien plus rapide que un POST par image."""
    safe, d = _sid_dir(sid)
    if not d or not d.exists():
        return JSONResponse({"ok": False, "error": "session inconnue"}, status_code=400)
    try:
        import numpy as np, cv2, json as _json
        data = file.file.read()
        ov = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if ov is None:
            return {"ok": False, "error": "overlay illisible"}
        try:
            ids = _json.loads(idxs or "[]")
        except Exception:
            ids = []
        n = 0
        for _idx in ids:
            try:
                base_p = d / ("f%06d.jpg" % int(_idx))
                if not base_p.exists():
                    continue
                base = cv2.imread(str(base_p))
                if base is None:
                    continue
                o = ov
                if o.shape[0] != base.shape[0] or o.shape[1] != base.shape[1]:
                    o = cv2.resize(o, (base.shape[1], base.shape[0]))
                if o.ndim == 3 and o.shape[2] == 4:
                    a = o[:, :, 3:4].astype(np.float32) / 255.0
                    fg = o[:, :, :3].astype(np.float32)
                    out = (fg * a + base.astype(np.float32) * (1.0 - a)).astype(np.uint8)
                else:
                    out = base
                cv2.imwrite(str(base_p), out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                n += 1
            except Exception:
                continue
        return {"ok": True, "n": n}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/render_overlay_store")
def render_overlay_store(sid: str = Form(...), idxs: str = Form(...), file: UploadFile = File(...)):
    """OPTIMISATION LOURDE : au lieu de composer chaque dessin sur l'image (OpenCV, une par une),
    on STOCKE simplement la surcouche PNG sous o{idx}.png. La composition sera faite EN UNE SEULE
    PASSE par ffmpeg à l'assemblage final (filtre overlay), bien plus rapide. Un même dessin partagé
    par plusieurs images (dessin statique) est écrit une fois puis lié en dur (instantané)."""
    safe, d = _sid_dir(sid)
    if not d or not d.exists():
        return JSONResponse({"ok": False, "error": "session inconnue"}, status_code=400)
    try:
        import json as _json
        data = file.file.read()
        ids = _json.loads(idxs or "[]")
        first = None
        for _idx in ids:
            p = d / ("o%06d.png" % int(_idx))
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
            if first is None:
                with open(p, "wb") as f:
                    f.write(data)
                first = p
            else:
                try:
                    os.link(str(first), str(p))   # lien dur = instantané, aucun espace disque
                except Exception:
                    with open(p, "wb") as f:
                        f.write(data)
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/autocal")
def autocal(vkey: str = Form(...), t: float = Form(0.0), minp: int = Form(15)):
    """Calibration AUTOMATIQUE best-effort : détecte la pelouse (masque vert), en extrait
    le quadrilatère du terrain et le met en correspondance avec le rectangle 105x68.
    Marche bien en vue large / caméra tactique ; renvoie ok:false en plan serré
    (l'app bascule alors sur la calibration manuelle guidée)."""
    vp = VKEYS.get(vkey)
    if not vp or not os.path.exists(vp):
        return JSONResponse({"error": "video_absente", "need_file": True}, status_code=409)
    try:
        import cv2, numpy as np
        cap = cv2.VideoCapture(vp); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0); ok, im = cap.read(); cap.release()
        if not ok or im is None:
            return {"ok": False, "reason": "frame"}
        H0, W0 = im.shape[:2]
        # PnLCalib (fort) en priorité, sinon repli YOLO-pose. Pas de repli "pelouse".
        ai = field_pts(im, minp)
        if ai and len(ai) >= 4:
            return {"ok": True, "method": "ai",
                    "pts": [{"f": p["f"], "i": p["i"]} for p in ai], "n": len(ai), "w": W0, "h": H0}
        return {"ok": False, "reason": "ai_insufficient"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/autocal_img")
async def autocal_img(file: UploadFile = File(...), minp: int = Form(15)):
    """Calibration AUTOMATIQUE sur une IMAGE fournie (ARRÊT SUR IMAGE) : PnLCalib (fort) + suppression
    du bleu, exactement comme /autocal mais sur l'image envoyée directement (pas besoin de la vidéo sur
    le serveur). Renvoie les points {f,i} normalisés (repère 105x68 -> image), ou ok:false si insuffisant."""
    try:
        import cv2, numpy as np
        data = await file.read()
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            return {"ok": False, "reason": "image"}
        H0, W0 = im.shape[:2]
        ai = field_pts(im, minp)
        if ai and len(ai) >= 4:
            return {"ok": True, "method": "ai",
                    "pts": [{"f": p["f"], "i": p["i"]} for p in ai], "n": len(ai), "w": W0, "h": H0}
        return {"ok": False, "reason": "ai_insufficient"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/snap_manual")
def snap_manual(vkey: str = Form(...), t: float = Form(0.0), corners: str = Form(...)):
    """RECALAGE LIGNES d'un mappage MANUEL : prend les 4 coins du terrain (normalisés 0..1,
    ordre (0,0)(105,0)(105,68)(0,68)) posés à la main et les colle aux lignes blanches réelles
    de l'image a l'instant t. Rend au manuel la précision du calibrage auto (terrain 105x68 exact,
    lignes alignées). Renvoie les coins affines (normalisés) + residu px, ou ok:false si pas de lignes."""
    vp = VKEYS.get(vkey)
    if not vp or not os.path.exists(vp):
        return JSONResponse({"error": "video_absente", "need_file": True}, status_code=409)
    try:
        import cv2, numpy as np, json as _json
        cs = _json.loads(corners)
        if not isinstance(cs, list) or len(cs) < 4:
            return {"ok": False, "reason": "corners"}
        cap = cv2.VideoCapture(vp)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, im = cap.read(); cap.release()
        if not ok or im is None:
            return {"ok": False, "reason": "frame"}
        H0, W0 = im.shape[:2]
        Ws = 1280 if W0 > 1280 else W0   # 1280 (au lieu de 960) : capte les lignes fines lointaines (Veo grand-angle)
        Hs = int(round(H0 * Ws / float(W0)))
        small = cv2.resize(im, (Ws, Hs)) if Ws != W0 else im
        dt = _white_dt(small)
        if dt is None:
            return {"ok": False, "reason": "no_lines"}   # aucune ligne blanche détectée -> on garde le manuel
        cpx = [[float(cs[k][0]) * Ws, float(cs[k][1]) * Hs] for k in range(4)]
        # 1) recalage rigide (glisse/pivote/zoome sur les lignes) puis 2) affinage ICP (perspective).
        snapped, res = line_snap(dt, cpx)
        if snapped is None:
            snapped = cpx
        refined, r2 = line_refine(dt, snapped, iters=6)
        fin = refined if (refined is not None and r2 is not None) else snapped
        rr = r2 if (refined is not None and r2 is not None) else res
        if not _quad_ok(fin, Ws, Hs):
            return {"ok": False, "reason": "degenere"}
        ni = [[round(fin[k][0] / Ws, 4), round(fin[k][1] / Hs, 4)] for k in range(4)]
        return {"ok": True, "corners": ni, "res": (round(float(rr), 2) if rr is not None else None)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/snap_manual_img")
async def snap_manual_img(file: UploadFile = File(...), corners: str = Form(...)):
    """RECALAGE LIGNES sur une IMAGE fournie (arrêt sur image) : mêmes calcul et précision que
    /snap_manual, mais sur une image envoyée directement (pas besoin de la vidéo sur le serveur).
    Reçoit l'image figée + 4 coins terrain normalisés -> renvoie les coins collés aux lignes blanches."""
    try:
        import cv2, numpy as np, json as _json
        cs = _json.loads(corners)
        if not isinstance(cs, list) or len(cs) < 4:
            return {"ok": False, "reason": "corners"}
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is None:
            return {"ok": False, "reason": "image"}
        H0, W0 = im.shape[:2]
        Ws = 1280 if W0 > 1280 else W0   # 1280 (au lieu de 960) : capte les lignes fines lointaines (Veo grand-angle)
        Hs = int(round(H0 * Ws / float(W0)))
        small = cv2.resize(im, (Ws, Hs)) if Ws != W0 else im
        dt = _white_dt(small)
        if dt is None:
            return {"ok": False, "reason": "no_lines"}
        cpx = [[float(cs[k][0]) * Ws, float(cs[k][1]) * Hs] for k in range(4)]
        snapped, res = line_snap(dt, cpx)
        if snapped is None:
            snapped = cpx
        refined, r2 = line_refine(dt, snapped, iters=6)
        fin = refined if (refined is not None and r2 is not None) else snapped
        rr = r2 if (refined is not None and r2 is not None) else res
        if not _quad_ok(fin, Ws, Hs):
            return {"ok": False, "reason": "degenere"}
        ni = [[round(fin[k][0] / Ws, 4), round(fin[k][1] / Hs, 4)] for k in range(4)]
        return {"ok": True, "corners": ni, "res": (round(float(rr), 2) if rr is not None else None)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/calibrate_clip")
def calibrate_clip(vkey: str = Form(...), t0: float = Form(0.0), t1: float = Form(0.0),
                   step: float = Form(0.6), minp: int = Form(15)):
    """Calibration IA sur TOUT le clip : échantillonne des images régulièrement et
    détecte les points-clés du terrain à chaque fois -> liste de points-clés (t, pts).
    La calibration suit ainsi les mouvements de caméra du début à la fin."""
    vp = VKEYS.get(vkey)
    if not vp or not os.path.exists(vp):
        return JSONResponse({"error": "video_absente", "need_file": True}, status_code=409)
    if get_field_model() is None:
        return {"keyframes": [], "n": 0, "reason": "model"}
    try:
        import cv2
        cap = cv2.VideoCapture(vp); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); dur = (total / fps) if fps else 0
        a = max(0.0, t0); b = t1 if (t1 and t1 > 0) else dur
        if b <= a: b = dur
        st = max(step, (b - a) / 80.0)  # borne ~80 échantillons max
        out = []; tt = a
        while tt <= b + 1e-3:
            cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000.0)
            ok, im = cap.read()
            if ok and im is not None:
                pts = field_pts(im, minp)
                if pts and len(pts) >= 4:
                    out.append({"t": round(tt, 2),
                                "pts": [{"f": p["f"], "i": p["i"]} for p in pts], "n": len(pts)})
            tt += st
        cap.release()
        return {"keyframes": out, "n": len(out), "dur": round(dur, 2)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/ai_debug")
def ai_debug(vkey: str = Form(...), t: float = Form(0.0)):
    """Diagnostic : renvoie les points-clés BRUTS détectés par le modèle (idx, conf, position)."""
    vp = VKEYS.get(vkey)
    if not vp or not os.path.exists(vp):
        return JSONResponse({"error": "video_absente"}, status_code=409)
    m = get_field_model()
    if m is None:
        return {"error": "model_absent"}
    try:
        import cv2, numpy as np
        cap = cv2.VideoCapture(vp); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, im = cap.read(); cap.release()
        if not ok:
            return {"error": "frame"}
        H, W = im.shape[:2]
        _kw = dict(conf=0.15, imgsz=getattr(m, "_imgsz", 1280), verbose=False)
        if not getattr(m, "_coreml", False):
            _kw["device"] = pick_device()
        r = m.predict(im, **_kw)[0]
        kp = getattr(r, "keypoints", None)
        if kp is None or kp.xy is None or len(kp.xy) == 0:
            return {"instances": 0}
        xy = kp.xy.cpu().numpy(); cf = kp.conf.cpu().numpy() if getattr(kp, "conf", None) is not None else None
        inst = []
        for i in range(xy.shape[0]):
            pts = []
            for k in range(xy.shape[1]):
                c = float(cf[i][k]) if cf is not None else 1.0
                pts.append({"i": k, "c": round(c, 2),
                            "x": round(float(xy[i][k][0]) / W, 3), "y": round(float(xy[i][k][1]) / H, 3)})
            inst.append({"n_conf04": sum(1 for p in pts if p["c"] >= 0.4),
                         "n_conf02": sum(1 for p in pts if p["c"] >= 0.2),
                         "pts": pts})
        return {"instances": xy.shape[0], "kpt": xy.shape[1], "data": inst}
    except Exception as e:
        return {"error": str(e)}

def _aph(H, x, y):
    d = (H[2][0] * x + H[2][1] * y + H[2][2]) or 1e-9
    return ((H[0][0] * x + H[0][1] * y + H[0][2]) / d,
            (H[1][0] * x + H[1][1] * y + H[1][2]) / d)

@app.post("/propagate_calib")
def propagate_calib(vkey: str = Form(...), t0: float = Form(0.0), t1: float = Form(0.0),
                    seed_t: float = Form(0.0), hf2i: str = Form(""), step: float = Form(0.3),
                    seeds: str = Form(""), snap: int = Form(1), from_t: float = Form(0.0)):
    """Propage UN calibrage manuel (homographie Hf2i au temps seed_t) sur tout le clip,
    en suivant le mouvement de caméra par flux optique. -> calibrage quasi parfait par frame
    sans dépendre du modèle de points-clés."""
    vp = VKEYS.get(vkey)
    if not vp or not os.path.exists(vp):
        return JSONResponse({"error": "video_absente", "need_file": True}, status_code=409)
    try:
        import cv2, numpy as np, json as _j
        H0 = None
        if hf2i:
            try: H0 = np.array(_j.loads(hf2i), dtype="float64").reshape(3, 3)
            except Exception: H0 = None
        cap = cv2.VideoCapture(vp); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); dur = (total / fps) if fps else 0
        a = max(0.0, t0); b = t1 if (t1 and t1 > 0) else dur
        if b <= a: b = dur
        # PROPAGATION PARTIELLE : si from_t est fourni, on ne suit la caméra et ne recalcule QUE
        # [from_t -> fin]. Le début (déjà calibré) n'est pas retouché -> beaucoup plus rapide.
        if from_t and from_t > a + 0.02:
            a = float(from_t)
            print("  propagation PARTIELLE depuis %.1fs" % a, flush=True)
        cap.set(cv2.CAP_PROP_POS_MSEC, a * 1000.0)
        ok, frame = cap.read()
        if not ok: return {"keyframes": [], "n": 0}
        Hh, Ww = frame.shape[:2]
        sc = 1920.0 / Ww if Ww > 1920 else 1.0   # suivi caméra à plus haute résolution = plus précis
        Ws, Hs = int(Ww * sc), int(Hh * sc)
        def small_gray(f):
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            return cv2.resize(g, (Ws, Hs)) if sc != 1.0 else g
        lk = dict(winSize=(21, 21), maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        _sub = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        def feats(g):
            # plus de points + qualité plus permissive = couverture dense (herbe, ombres, structures),
            # même quand on ne voit pas les lignes -> suivi caméra plus robuste.
            f = cv2.goodFeaturesToTrack(g, maxCorners=3000, qualityLevel=0.006, minDistance=6, blockSize=7)
            if f is not None:
                try: cv2.cornerSubPix(g, f, (5, 5), (-1, -1), _sub)
                except Exception: pass
            return f
        prev = small_gray(frame)
        p0 = feats(prev)
        Hcum = np.eye(3); times = [a]; cums = [Hcum.copy()]; mot = [0.0]
        cn = np.array([[[0, 0], [Ws, 0], [Ws, Hs], [0, Hs]]], dtype="float32")
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            fi += 1; t = a + fi / fps
            if t > b + 1e-3: break
            cur = small_gray(frame)
            Hstep = np.eye(3); mstep = 0.0
            if p0 is not None and len(p0) >= 8:
                p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **lk)
                # COHÉRENCE AVANT/ARRIÈRE : on retrace p1 vers l'image précédente ; on ne garde que les
                # points qui reviennent à leur position d'origine (<1 px) -> élimine joueurs en mouvement,
                # reflets, points instables. C'est ce qui rend le suivi caméra précis et sans dérive.
                if p1 is not None:
                    p0r, st0, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **lk)
                    fb = np.abs(p0 - p0r).reshape(-1, 2).max(axis=1)
                    good = (st1.flatten() == 1) & (st0.flatten() == 1) & (fb < 1.0)
                    gp = p0[good]; gc = p1[good]
                    if len(gp) >= 12:
                        mstep = float(np.median(np.linalg.norm(gc - gp, axis=1)))
                        # VERROU CAMÉRA IMMOBILE : si le déplacement médian est minuscule (< 0,45 px), on
                        # FORCE l'identité -> AUCUN cumul de bruit, le terrain ne bouge pas du tout quand
                        # la caméra ne bouge pas. (Un vrai pan/zoom est bien plus grand, donc préservé.)
                        if mstep >= 0.45:
                            Hs2, inl = cv2.findHomography(gp, gc, cv2.RANSAC, 2.0, maxIters=4000, confidence=0.999)
                            if Hs2 is not None:
                                pr = cv2.perspectiveTransform(cn, Hs2.astype("float32"))[0]
                                # Un PAS image->image doit être PETIT (une caméra ne saute pas). On rejette
                                # tout pas > 4% de l'image -> plus de dérive/sauts qui envoient le terrain
                                # n'importe où. (Un vrai pan reste bien en dessous, image par image.)
                                if float(np.linalg.norm(pr - cn[0], axis=1).max()) < 0.04 * max(Ws, Hs):
                                    Hstep = Hs2
            Hcum = Hstep @ Hcum
            times.append(t); cums.append(Hcum.copy()); mot.append(mstep)
            # RE-DÉTECTION À CHAQUE IMAGE : des points neufs et denses à chaque pas -> l'estimation
            # image→image est la plus précise possible, donc la dérive cumulée reste minimale.
            p0 = feats(cur)
            prev = cur
        cap.release()
        corners_f = [(0, 0), (105, 0), (105, 68), (0, 68)]
        # GRAINES = tes calibrages manuels (1 ou plusieurs). Chaque graine devient une ANCRE locale :
        # pour chaque image, on propage depuis la graine la PLUS PROCHE -> tes corrections sont prises
        # en compte localement, pas écrasées.
        seed_list = []
        try:
            if seeds:
                for s in _j.loads(seeds):
                    Hm = np.array(s["hf2i"], dtype="float64").reshape(3, 3); st = float(s["t"])
                    seed_list.append((st, Hm))
        except Exception:
            seed_list = []
        if not seed_list and H0 is not None:
            seed_list = [(seed_t, H0)]
        # PROPAGATION PARTIELLE : on ne garde que les graines DANS la plage recalculée [a-0.6, b+0.6]
        # (l'ancre à from_t sert de départ ; les graines d'avant ne concernent pas cette portion).
        seed_list = [(st, Hm) for (st, Hm) in seed_list if (a - 0.6) <= st <= (b + 0.6)]
        if not seed_list:
            return JSONResponse({"error": "aucune graine de calibrage"}, status_code=400)
        seeds_px = []  # (t, idx, corners_px(1,4,2), inv_cum)
        for (st, Hm) in seed_list:
            spx = np.array([[[_aph(Hm, fx, fy)[0] * Ws, _aph(Hm, fx, fy)[1] * Hs] for (fx, fy) in corners_f]], dtype="float32")
            sidx = min(range(len(times)), key=lambda i: abs(times[i] - st))
            seeds_px.append((st, sidx, spx, np.linalg.inv(cums[sidx])))
        # ÉMISSION ADAPTATIVE : dense quand la caméra bouge, RARE quand elle est statique.
        MOT_STEP = 9.0; MIN_DT = 0.12; MAX_DT = 1.6   # + dense en mouvement -> terrain qui suit + vite la caméra
        emit_idx = [0]; acc = 0.0; last_t = times[0]
        for i in range(1, len(times)):
            acc += mot[i]; dtt = times[i] - last_t
            if dtt < MIN_DT: continue
            if acc >= MOT_STEP or dtt >= MAX_DT:
                emit_idx.append(i); last_t = times[i]; acc = 0.0
        if emit_idx[-1] != len(times) - 1: emit_idx.append(len(times) - 1)
        for (_st, _sidx, _spx, _sinv) in seeds_px:
            if _sidx not in emit_idx: emit_idx.append(_sidx)
        emit_idx = sorted(set(emit_idx))
        seeds_sorted = sorted(seeds_px, key=lambda s: s[0])
        def prop_corners(seed, i):
            st_, sidx_, spx_, sinv_ = seed
            cc = cv2.perspectiveTransform(spx_, (cums[i] @ sinv_).astype("float32"))[0]
            return np.array([[float(cc[k][0]), float(cc[k][1])] for k in range(4)], dtype="float64")
        # RECALAGE LIGNES pendant la propagation : on recolle le terrain sur les VRAIES lignes blanches
        # à chaque image émise (comme le mappage manuel) -> corrige la dérive du flux optique, le terrain
        # RESTE EN PLACE. Accepté SEULEMENT si fiable (petit résidu) pour ne jamais introduire de saut.
        _snapcap = cv2.VideoCapture(vp) if snap else None
        _snap_ok = 0
        out = []; prev_cp = None
        for i in emit_idx:
            t = times[i]
            # ancres encadrantes : la dernière avant t (lo) et la première après t (hi)
            lo = None; hi = None
            for s in seeds_sorted:
                if s[0] <= t + 1e-6 and (lo is None or s[0] > lo[0]): lo = s
                if s[0] >= t - 1e-6 and (hi is None or s[0] < hi[0]): hi = s
            if lo and hi and lo is not hi and hi[0] > lo[0]:
                gap = hi[0] - lo[0]
                r = (t - lo[0]) / gap
                if gap <= 2.0:
                    # ANCRES RAPPROCHÉES (méthode LiveTag) : entre deux calages manuels proches, on
                    # INTERPOLE EN LIGNE DROITE tes positions de coins -> transition PARFAITEMENT LISSE,
                    # sans le moindre saut du flux optique. Passe exactement par chaque ancre.
                    A = lo[2][0]; B = hi[2][0]
                    cp = A * (1 - r) + B * r
                else:
                    # ancres éloignées : on suit la caméra (flux optique) et on répartit la dérive.
                    A = prop_corners(lo, i); B = prop_corners(hi, i)
                    cp = A * (1 - r) + B * r
            else:
                cp = prop_corners(lo or hi, i)
            corners_px = [[float(cp[k][0]), float(cp[k][1])] for k in range(4)]
            dnear = min(abs(t - s[0]) for s in seeds_sorted)
            near_seed = dnear < 0.18
            # recalage lignes (hors des images-ancres manuelles, qui sont déjà la vérité)
            if _snapcap is not None and not near_seed:
                try:
                    _snapcap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                    _okf, _fr = _snapcap.read()
                    if _okf and _fr is not None:
                        _sm = cv2.resize(_fr, (Ws, Hs)) if _fr.shape[1] != Ws else _fr
                        _dt = _white_dt(_sm)
                        if _dt is not None:
                            _snp, _res = line_snap(_dt, corners_px)
                            # On n'accepte le recalage QUE si c'est un PETIT ajustement (raffinement),
                            # jamais une relocalisation : sinon le terrain « saute » sur une mauvaise ligne.
                            if _snp is not None and _res is not None and _res <= 3.0 and _quad_ok(_snp, Ws, Hs):
                                try:   # AFFINAGE PERSPECTIF (ICP) : colle toutes les lignes du modèle au pixel près
                                    _rc, _rr = line_refine(_dt, _snp, iters=12)
                                    if _rc is not None and _rr is not None and _rr <= _res + 0.3 and _quad_ok(_rc, Ws, Hs): _snp = _rc; _res = _rr
                                except Exception: pass
                                _mv = max(((_snp[k][0] - corners_px[k][0]) ** 2 + (_snp[k][1] - corners_px[k][1]) ** 2) ** 0.5 for k in range(4))
                                if _mv < 0.05 * max(Ws, Hs):
                                    corners_px = [[float(_snp[k][0]), float(_snp[k][1])] for k in range(4)]
                                    _snap_ok += 1
                except Exception:
                    pass
            # ANTI-GLISSEMENT EN STATIQUE : si le terrain n'a quasiment pas bougé depuis l'image-clé
            # précédente (≤ 8 px), on le FIGE à l'identique -> plus de tremblement/glissement quand la
            # caméra est immobile. (N'affecte pas les vrais déplacements, bien plus grands.)
            if prev_cp is not None and not near_seed:
                md = max(((corners_px[k][0] - prev_cp[k][0]) ** 2 + (corners_px[k][1] - prev_cp[k][1]) ** 2) ** 0.5 for k in range(4))
                if md < 8.0: corners_px = [list(c) for c in prev_cp]
            prev_cp = [list(c) for c in corners_px]
            # CONFIANCE HONNÊTE : proximité d'une ancre manuelle (proche d'un calage à la main = fiable).
            conf = round(max(0.3, 1.0 - 0.30 * dnear), 2)
            ni = [[corners_px[k][0] / Ws, corners_px[k][1] / Hs] for k in range(4)]
            pts = [{"f": list(corners_f[k]), "i": [round(ni[k][0], 4), round(ni[k][1], 4)]} for k in range(4)]
            # ok=True partout : on NE flague PLUS automatiquement des images « à corriger ». Seules tes
            # ancres manuelles s'affichent comme repères ; tu corriges visuellement où TU vois un écart.
            out.append({"t": round(t, 2), "pts": pts, "conf": conf,
                        "ok": True, "kind": ("anchor" if near_seed else "prop")})
        if _snapcap is not None:
            _snapcap.release()
            print("  propagation : recalage lignes appliqué sur %d/%d images émises" % (_snap_ok, len(emit_idx)), flush=True)
        return {"keyframes": out, "n": len(out)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/pnl_test")
def pnl_test(t: float = 2.0):
    """Test inférence PnLCalib sur une vidéo en cache (1re trouvée)."""
    import glob
    vids = sorted(glob.glob(str(VID / "*.mp4")) + glob.glob(str(VID / "*.mov")))
    if not vids: return {"error": "aucune vidéo en cache"}
    try:
        import cv2
        cap = cv2.VideoCapture(vids[0]); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, im = cap.read(); cap.release()
        if not ok: return {"error": "frame"}
        pts = pnl_field_pts(im)
        return {"video": os.path.basename(vids[0]), "ok": bool(pts),
                "pts": pts if pts else None}
    except Exception as e:
        import traceback; traceback.print_exc(); return {"error": str(e)}

@app.get("/pnl_warm")
def pnl_warm():
    """Force le téléchargement + chargement de PnLCalib (1re fois, peut être long)."""
    ok = bool(get_pnl())
    return {"loaded": ok,
            "pnl_kp_mb": round(os.path.getsize(PNL_KP_W) / 1e6, 1) if PNL_KP_W.exists() else 0,
            "pnl_lines_mb": round(os.path.getsize(PNL_LINES_W) / 1e6, 1) if PNL_LINES_W.exists() else 0}

_CALIB_CACHE = {}   # VITESSE : on mémorise le résultat de l'auto-calibrage par (clip, instants, params)
_CALIB_PROG = {}    # pid -> {"p":pourcentage, "msg":texte} : progression live de l'auto-calibrage
def _cp(pid, p, msg=""):
    if pid: _CALIB_PROG[pid] = {"p": int(max(0, min(100, p))), "msg": msg}

@app.get("/calib_progress")
def calib_progress(pid: str = ""):
    return _CALIB_PROG.get(pid, {"p": 0, "msg": ""})

def _quad_ok(q, Ws, Hs):
    """Un quadrilatère de terrain (4 coins px, ordre TL,TR,BR,BL) est-il SAIN ?
    Rejette les homographies dégénérées (terrain replié/croisé/minuscule) qui font
    "converger toutes les lignes" -> empêche l'overlay de sauter/s'effondrer."""
    import numpy as np
    P = np.array(q, dtype="float64")
    if P.shape != (4, 2) or not np.isfinite(P).all():
        return False
    # aire significative (anti-effondrement "toutes les lignes se rejoignent")
    area = 0.0
    for k in range(4):
        x1, y1 = P[k]; x2, y2 = P[(k + 1) % 4]; area += x1 * y2 - x2 * y1
    if abs(area) / 2.0 < 0.02 * Ws * Hs:
        return False
    # convexité stricte : tous les produits vectoriels de même signe (sinon quad croisé)
    sgn = 0
    for k in range(4):
        a = P[(k + 1) % 4] - P[k]; b = P[(k + 2) % 4] - P[(k + 1) % 4]
        cr = a[0] * b[1] - a[1] * b[0]
        s = 1 if cr > 0 else -1
        if sgn == 0: sgn = s
        elif s != sgn: return False
    # aucun côté quasi nul
    for k in range(4):
        if float(np.linalg.norm(P[(k + 1) % 4] - P[k])) < 0.05 * max(Ws, Hs):
            return False
    return True


def _quad_proportions_ok(q):
    """Rejette une ANCRE géométriquement invraisemblable : typiquement PnL n'a vu qu'une moitié de
    terrain -> la touche lointaine est extrapolée de travers (proportions fausses, 'touche fantôme').
    Seuils LARGES : ne bloque que les cas gros, jamais une vraie perspective broadcast.
    q = [TL, TR, BR, BL] pour le terrain 105x68."""
    import numpy as np
    P = np.array(q, dtype="float64")
    e_len = [float(np.linalg.norm(P[0] - P[1])), float(np.linalg.norm(P[2] - P[3]))]  # 2 côtés 105 m
    e_wid = [float(np.linalg.norm(P[1] - P[2])), float(np.linalg.norm(P[3] - P[0]))]  # 2 côtés 68 m
    if min(e_len + e_wid) < 3.0:
        return False
    # un côté parallèle ne doit pas être écrasé > 8x par rapport à son opposé (perspective réaliste)
    if max(e_len) / max(1e-6, min(e_len)) > 8.0:
        return False
    if max(e_wid) / max(1e-6, min(e_wid)) > 8.0:
        return False
    # rapport longueur/largeur cohérent avec 105/68 ≈ 1.54, avec grande marge de perspective
    L = 0.5 * (e_len[0] + e_len[1]); W = 0.5 * (e_wid[0] + e_wid[1])
    r = L / max(1e-6, W)
    if not (0.45 <= r <= 6.0):
        return False
    return True

def _terrain_plausible(q, Ws, Hs):
    """Rejette un TERRAIN projeté géométriquement aberrant (vue basse/large type Veo : l'extrapolation
    du terrain lointain explose ou s'inverse). q = [TL,TR,BR,BL] en pixels. Complète _quad_ok :
    - perspective inversée (touche lointaine plus large que la proche) = homographie retournée ;
    - terrain démesurément large (touche proche > 9x l'image) = échelle partie en vrille ;
    - hauteur aberrante. Ces cas produisent le 'terrain qui grandit/rétrécit / lignes en biais'."""
    import numpy as np
    P = np.array(q, dtype="float64")
    if P.shape != (4, 2) or not np.isfinite(P).all():
        return False
    wtop = abs(P[1][0] - P[0][0]) / max(1.0, Ws)   # largeur touche lointaine (haut)
    wbot = abs(P[2][0] - P[3][0]) / max(1.0, Ws)   # largeur touche proche (bas)
    hgt = abs(((P[3][1] + P[2][1]) / 2) - ((P[0][1] + P[1][1]) / 2)) / max(1.0, Hs)
    if wtop > wbot * 1.02:   # la touche lointaine ne peut pas être plus large que la proche
        return False
    if wbot > 9.0:           # terrain démesurément large -> homographie dégénérée
        return False
    if hgt > 1.8 or hgt < 0.05:
        return False
    return True

@app.post("/calibrate_smart")
def calibrate_smart(vkey: str = Form(...), t0: float = Form(0.0), t1: float = Form(0.0),
                    anchor_dt: float = Form(4.0), minp: int = Form(15), pid: str = Form("")):
    """Calibration façon LiveTag : 1) suivi du mouvement de caméra (flux optique) sur tout le
    clip, 2) PnLCalib sur quelques ANCRES précises, 3) propagation de la calibration le long
    du mouvement réel + points-clés ADAPTATIFS (serrés quand la caméra bouge)."""
    print("RECALAGE demandé — vkey=%s  t0=%.1f t1=%.1f" % (vkey, t0, t1), flush=True)
    vp = _resolve_vid(vkey)   # ignore .part / _win_ ; retrouve le vrai match même après redémarrage
    if not vp or not os.path.exists(vp):
        print("  -> vidéo introuvable côté serveur", flush=True)
        return JSONResponse({"error": "video_absente", "need_file": True}, status_code=409)
    print("  -> source :", os.path.basename(vp), flush=True)
    # RAPIDE : on ne prépare (image-par-image) QUE la fenêtre du clip, pas les 47 min. Pour un MP4
    # téléchargé -> renvoyé tel quel (T_OFF=0). Pour un WebM direct -> petite fenêtre en ~quelques s.
    _cp(pid, 1, "préparation du clip…")
    try:
        vp, T_OFF = _seekable_for_window(vp, t0, t1, margin=3.0)   # temps locaux = temps d'origine - T_OFF
    except Exception as _pe:
        print("prep window err:", _pe, flush=True); T_OFF = 0.0
    print("  -> prêt : %s  (t_off=%.1fs)" % (os.path.basename(vp), T_OFF), flush=True)
    try:
        import cv2, numpy as np, time as _time
        _tA = _time.perf_counter(); _anc_errs = []; _snap_res = []
        _cdiag("\n===== NOUVEAU CALIBRAGE  vkey=%s  t0=%.1f t1=%.1f =====" % (os.path.basename(vp), t0, t1))
        _cp(pid, 2, "préparation…")
        cap = cv2.VideoCapture(vp); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); dur = (total / fps) if fps else 0
        a = max(0.0, t0 - T_OFF); b = (t1 - T_OFF) if (t1 and t1 > 0) else dur   # temps LOCAUX (fenêtre)
        if b <= a: b = dur
        _nfr = max(1, int((b - a) * fps))
        print("  cap ouvert : fps=%.2f total=%d dur=%.0fs  clip local[%.1f, %.1f]" % (fps, total, dur, a, b), flush=True)
        cap.set(cv2.CAP_PROP_POS_MSEC, a * 1000.0); ok, frame = cap.read()
        if not ok:
            print("  -> lecture 1re image ÉCHOUÉE à %.1fs" % a, flush=True)
            return {"keyframes": [], "n": 0}
        print("  1re image lue OK -> suivi caméra…", flush=True)
        Hh, Ww = frame.shape[:2]; sc = 1440.0 / Ww if Ww > 1440 else 1.0   # 1440 (au lieu de 960) : recalage lignes + précis
        Ws, Hs = int(Ww * sc), int(Hh * sc)
        def sg(f):
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); return cv2.resize(g, (Ws, Hs)) if sc != 1.0 else g
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        def feats(g): return cv2.goodFeaturesToTrack(g, maxCorners=1200, qualityLevel=0.006, minDistance=6)
        prev = sg(frame); p0 = feats(prev)
        # Taux de PELOUSE (vert) par image -> sert à savoir si le TERRAIN est visible (sinon = public/gros plan).
        def _grn(f):
            gw = 128; gh = max(1, int(gw * f.shape[0] / max(1, f.shape[1])))
            gs = cv2.resize(f, (gw, gh)); hv = cv2.cvtColor(gs, cv2.COLOR_BGR2HSV)
            gm = cv2.inRange(hv, (30, 25, 25), (95, 255, 255))
            return float(int(gm.sum()) // 255) / float(gw * gh)
        Hcum = np.eye(3); times = [a]; cums = [Hcum.copy()]; mot = [0.0]; grn = [_grn(frame)]; fi = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            fi += 1; t = a + fi / fps
            if t > b + 1e-3: break
            if pid and fi % 4 == 0: _cp(pid, 4 + int(34 * min(1.0, fi / _nfr)), "suivi caméra…")
            cur = sg(frame); Hstep = np.eye(3); m = 0.0
            if p0 is not None and len(p0) >= 8:
                p1, stt, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **lk)
                if p1 is not None:
                    gp = p0[stt == 1]; gc = p1[stt == 1]
                    if len(gp) >= 6:
                        # MODÈLE CAMÉRA = SIMILITUDE (translation + rotation + zoom UNIFORME, 4 ddl).
                        # PAS de perspective ni de cisaillement -> une ligne droite reste droite, le
                        # terrain ne "gonfle" pas et la ligne médiane ne part pas en biais sous un
                        # zoom / pan latéral (une homographie complète, elle, ajoutait ces déformations).
                        _M2, inl = cv2.estimateAffinePartial2D(gp, gc, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                        Hs2 = np.vstack([_M2, [0.0, 0.0, 1.0]]) if _M2 is not None else None
                        if Hs2 is not None:
                            # GARDE-FOU : on rejette une homographie aberrante (saut/perspective
                            # impossible entre 2 images consécutives) pour ne jamais réexploser.
                            corners = np.array([[[0, 0], [Ws, 0], [Ws, Hs], [0, Hs]]], dtype="float32")
                            proj = cv2.perspectiveTransform(corners, Hs2.astype("float32"))[0]
                            moved = float(np.linalg.norm(proj - corners[0], axis=1).max())
                            if moved < 0.18 * max(Ws, Hs):
                                Hstep = Hs2.astype("float64")
                        m = float(np.median(np.linalg.norm(gc - gp, axis=1))) if len(gc) else 0.0
                        p0 = gc.reshape(-1, 1, 2)
            Hcum = Hstep @ Hcum; times.append(t); cums.append(Hcum.copy()); mot.append(m); grn.append(_grn(frame))
            if p0 is None or len(p0) < 120:
                p0 = feats(cur)
            prev = cur
        cap.release(); _tMot = _time.perf_counter()
        n = len(times)
        print("  suivi caméra OK : %d images en %.1fs -> détection ancres terrain…" % (n, _tMot - _tA), flush=True)
        # PLAGES « TERRAIN VISIBLE » : intervalles de temps où la pelouse couvre assez l'image.
        # Hors de ces plages (public, gros plan sans terrain) le client n'affichera PAS le terrain.
        GTH = 0.14
        try:
            _vis = [((grn[i] if i < len(grn) else 0.0) >= GTH) for i in range(n)]
            _rg = []; _s = None
            for i in range(n):
                if _vis[i] and _s is None: _s = times[i]
                elif (not _vis[i]) and _s is not None: _rg.append([_s, times[i]]); _s = None
            if _s is not None: _rg.append([_s, times[-1]])
            _mg = []
            for r in _rg:
                if _mg and r[0] - _mg[-1][1] < 0.6: _mg[-1][1] = r[1]
                else: _mg.append([r[0], r[1]])
            field_ranges = [[round(x + T_OFF, 2), round(y + T_OFF, 2)] for x, y in _mg if (y - x) >= 0.4]
        except Exception:
            field_ranges = []
        corners_f = [(0, 0), (105, 0), (105, 68), (0, 68)]
        # ANCRAGE PILOTÉ PAR LE MOUVEMENT CAMÉRA (logique LiveTag) : on pose une ancre dès que la
        # caméra a bougé de MOT_STEP depuis la dernière. Donc PEU d'ancres si la caméra est stable,
        # BEAUCOUP (très rapprochées) si elle panote/tourne vite. Sécurités : au moins MIN_DT entre
        # deux ancres, et au moins une toutes les MAX_DT s même caméra immobile.
        # MODE RAPIDE : ancres un peu plus espacées (moins d'appels réseau coûteux) sans perdre les pans.
        # PAS DE PLAFOND FIXE : le nombre d'ancres suit le MOUVEMENT (une ancre tous les MOT_STEP px de
        # mouvement caméra cumulé) -> dense en pan/zoom, rare en caméra calme. Un clip long en aura donc plus,
        # proportionnellement au mouvement. Bornes : au moins MIN_DT entre 2 ancres, au moins une toutes les MAX_DT s.
        MOT_STEP = 26.0; MIN_DT = 0.3; MAX_DT = 3.0
        idxs = [0]; acc = 0.0; last_t = times[0]
        for i in range(1, n):
            acc += mot[i]; dt = times[i] - last_t
            if dt < MIN_DT: continue
            if acc >= MOT_STEP or dt >= MAX_DT:
                idxs.append(i); last_t = times[i]; acc = 0.0
        if idxs and idxs[-1] != n - 1: idxs.append(n - 1)
        # (plus de plafond : l'émission ci-dessus est déjà pilotée par le mouvement caméra -> le nombre
        #  d'ancres est proportionnel au mouvement, donc un clip long/mouvementé en a naturellement plus.)
        cap2 = cv2.VideoCapture(vp); anchors = []; anc_idx = set(); _prop_rej = 0; _ai = 0
        print("  %d ancres à analyser (modèle terrain)…" % len(idxs), flush=True)
        for idx in idxs:
            _ai += 1
            if pid: _cp(pid, 40 + int(48 * _ai / max(1, len(idxs))), "ancres IA %d/%d…" % (_ai, len(idxs)))
            cap2.set(cv2.CAP_PROP_POS_MSEC, times[idx] * 1000.0); ok, fr = cap2.read()
            if not ok: continue
            globals()["_PNL_ERR"] = None
            # ANCRES : pas de raffinement lignes ici — la passe de recalage lignes dédiée (ci-dessous)
            # recale déjà chaque image sur les lignes blanches. On évite ainsi de doubler le travail
            # (gain ~plusieurs secondes par ancre).
            _ta0 = _time.perf_counter()
            pts = field_pts(fr, minp, refine=False)
            print("    ancre %d/%d @%.1fs -> %d pts (%.1fs)" % (_ai, len(idxs), times[idx] + T_OFF, len(pts or []), _time.perf_counter() - _ta0), flush=True)
            if pts and len(pts) >= 4:
                err = globals().get("_PNL_ERR")
                if err is not None: _anc_errs.append(float(err))
                aconf = max(0.05, min(1.0, 1.0 - (err if err is not None else 8.0) / 30.0))
                src = np.array([list(corners_f[k]) for k in range(4)], dtype="float32")
                dst = np.array([[pts[k]["i"][0], pts[k]["i"][1]] for k in range(4)], dtype="float32")
                Hf = cv2.getPerspectiveTransform(src, dst)  # terrain(m) -> image normalisée
                cs = np.array([[[float(_aph(Hf, c[0], c[1])[0]) * Ws,
                                 float(_aph(Hf, c[0], c[1])[1]) * Hs] for c in corners_f]], dtype="float32")
                if _quad_ok(cs[0], Ws, Hs) and _terrain_plausible(cs[0], Ws, Hs):
                    anchors.append((idx, cs, aconf)); anc_idx.add(idx)
                elif _quad_ok(cs[0], Ws, Hs):
                    _cdiag("    ancre @%.1fs REJETÉE (terrain aberrant : inversé/démesuré)" % (times[idx] + T_OFF))
        cap2.release(); _tAnc = _time.perf_counter()
        _cdiag("  ancres terrain VALIDES : %d / %d  (si 0 -> terrain non reconnu)" % (len(anchors), len(idxs)))
        if not anchors:
            print("  -> AUCUNE ancre valide : le terrain n'est pas reconnu sur ce clip -> recalage lignes NON lancé (calibrage auto impossible, passe en manuel).", flush=True)
            return {"keyframes": [], "n": 0, "reason": "pnl_fail"}
        def near_anchor(i):
            best = anchors[0]; bd = abs(anchors[0][0] - i)
            for an in anchors:
                d = abs(an[0] - i)
                if d < bd: bd = d; best = an
            return best
        # CHEMIN CAMÉRA (dense) : positions normalisées de 4 points de référence suivis par la caméra,
        # échantillonnées toutes ~0.15 s. Le client place le terrain image par image en appliquant ce
        # chemin depuis l'ancre PnL la plus proche -> le terrain suit les pans (horizontaux/verticaux).
        R0 = np.array([[[0.2 * Ws, 0.2 * Hs], [0.8 * Ws, 0.2 * Hs],
                        [0.8 * Ws, 0.8 * Hs], [0.2 * Ws, 0.8 * Hs]]], dtype="float32")
        cam = []; last_c = -1e9
        for i in range(n):
            if not (i == 0 or i == n - 1 or (times[i] - last_c) >= 0.15): continue
            last_c = times[i]
            rr = cv2.perspectiveTransform(R0, cums[i].astype("float32"))[0]
            cam.append({"t": round(times[i] + T_OFF, 3),
                        "r": [[round(float(rr[k][0]) / Ws, 5), round(float(rr[k][1]) / Hs, 5)] for k in range(4)]})
        # ÉMISSION DENSE + RECALAGE LIGNES : un point-clé toutes les ~0,3 s (et à chaque ancre). Chaque
        # point-clé = position propagée par le suivi caméra depuis l'ancre la plus proche, PUIS recalée
        # sur les lignes blanches réelles de CETTE image -> le terrain colle aux lignes tout du long,
        # plus seulement aux ancres.
        # images à recaler : DENSES pendant le mouvement, RARES quand c'est stable (le terrain n'y
        # bouge pas), plafonnées pour borner le temps (chaque image 4K coûte ~0,3 s à lire).
        # PAS DE PLAFOND : nombre d'états piloté par le MOUVEMENT (un état tous les SNAP_STEP px de mouvement
        # caméra) -> dense en pan/zoom, rare en caméra calme, proportionnel à la durée/au mouvement du clip.
        SNAP_STEP = 14.0; SNAP_MIN_DT = 0.18; SNAP_MAX_DT = 1.2   # densité motion-pilotée (bornée par MIN_DT)
        eidx = [0]; acc = 0.0; last_e = times[0]
        for i in range(1, n):
            acc += mot[i]; dtt = times[i] - last_e
            if dtt < SNAP_MIN_DT: continue
            if acc >= SNAP_STEP or dtt >= SNAP_MAX_DT:
                eidx.append(i); last_e = times[i]; acc = 0.0
        if eidx[-1] != n - 1: eidx.append(n - 1)
        emit_idx = sorted(set(eidx) | set(anc_idx))
        print("  recalage lignes : %d images à traiter…" % len(emit_idx), flush=True)
        cap4 = cv2.VideoCapture(vp); out = []; _degen = 0; _ei = 0
        for i in emit_idx:
            _ei += 1
            if pid: _cp(pid, 90 + int(9 * _ei / max(1, len(emit_idx))), "recalage lignes…")
            cap4.set(cv2.CAP_PROP_POS_MSEC, times[i] * 1000.0); okf, fr = cap4.read()
            dt = None
            if okf:
                small = cv2.resize(fr, (Ws, Hs)) if fr.shape[1] != Ws else fr
                dt = _white_dt(small)
            # CANDIDATS : les 2 ancres qui encadrent le mieux l'image. On propage depuis chacune, on
            # recale sur les lignes, et on GARDE la meilleure (résidu le plus faible) -> corrige la dérive
            # héritée d'un seul côté (médiane en biais / touche qui déborde sur les images faibles).
            corners_px = None; conf = 0.0; _bres = None; _best_cpx0 = None
            for (aidx, cs, aconf) in sorted(anchors, key=lambda an: abs(an[0] - i))[:2]:
                Hrel = (cums[i] @ np.linalg.inv(cums[aidx])).astype("float32")
                cc = cv2.perspectiveTransform(cs, Hrel)[0]
                cpx0 = [[float(cc[k][0]), float(cc[k][1])] for k in range(4)]   # propagation depuis l'ancre PnL (déjà fiable)
                cpx = cpx0; cf = round(aconf, 2); rr = None
                if dt is not None:
                    snapped, res = line_snap(dt, cpx0)
                    # RECALAGE LIGNES accepté SEULEMENT s'il est fiable (petit résidu). Sinon on GARDE la
                    # propagation PnL. Sur un direct (lignes moins nettes / bruit), un mauvais snap ne casse
                    # donc plus le calibrage : on émet quand même le terrain issu des ancres.
                    if snapped is not None and res is not None and res <= 5.0 and _quad_ok(snapped, Ws, Hs):
                        # AFFINAGE PERSPECTIF (ICP) après le recalage rigide : ajuste l'homographie COMPLÈTE
                        # pour coller TOUTES les lignes du modèle au pixel près (médiane, rond central, surfaces)
                        # -> corrige la petite erreur de perspective que le recalage rigide ne peut pas rattraper.
                        try:
                            _rc, _rr = line_refine(dt, snapped, iters=12)
                            if _rc is not None and _rr is not None and _rr <= res + 0.3 and _quad_ok(_rc, Ws, Hs):
                                snapped = _rc; res = _rr
                        except Exception:
                            pass
                        cpx = snapped; rr = float(res); cf = round(max(0.6, min(1.0, 1.0 - res / 12.0)), 2)
                if corners_px is None or cf > conf:
                    corners_px = cpx; conf = cf; _bres = rr; _best_cpx0 = cpx0
            if _bres is not None: _snap_res.append(_bres)
            if corners_px is None or not _quad_ok(corners_px, Ws, Hs) or not _terrain_plausible(corners_px, Ws, Hs):
                _degen += 1; continue   # terrain écrasé/dégénéré/aberrant : on n'émet PAS (le client interpole entre les bonnes frames)
            ni = [[corners_px[k][0] / Ws, corners_px[k][1] / Hs] for k in range(4)]
            pts = [{"f": list(corners_f[k]), "i": [round(ni[k][0], 4), round(ni[k][1], 4)]} for k in range(4)]
            _pre = None
            try:
                if _best_cpx0 is not None:
                    _p0=[[_best_cpx0[k][0]/Ws,_best_cpx0[k][1]/Hs] for k in range(4)]
                    _pre=[round(abs(_p0[1][0]-_p0[0][0]),2), round(abs(_p0[2][0]-_p0[3][0]),2), round(abs(((_p0[3][1]+_p0[2][1])/2)-((_p0[0][1]+_p0[1][1])/2)),2)]
            except Exception: _pre=None
            out.append({"t": round(times[i] + T_OFF, 2), "pts": pts, "conf": conf,
                        "ok": bool(conf >= 0.7), "kind": ("anchor" if i in anc_idx else "snap"), "_pre": _pre})
        cap4.release(); _tSnap = _time.perf_counter()
        out.sort(key=lambda k: k["t"])
        # DIAG terrain : largeur haute (touche lointaine), largeur basse (touche proche) et hauteur du
        # terrain projeté, par frame -> si ça varie beaucoup d'une frame à l'autre = terrain qui "saute".
        try:
            _cdiag("  --- terrain par frame : APRES recalage (haut/bas/hauteur) | AVANT recalage (propagation pure) ---")
            for k in out:
                _p = k["pts"]  # ordre TL,TR,BR,BL en coord terrain ; i = [x,y] normalisés image
                _xTL,_yTL=_p[0]["i"]; _xTR,_yTR=_p[1]["i"]; _xBR,_yBR=_p[2]["i"]; _xBL,_yBL=_p[3]["i"]
                _wtop=abs(_xTR-_xTL); _wbot=abs(_xBR-_xBL); _hgt=abs(((_yBL+_yBR)/2)-((_yTL+_yTR)/2))
                _pr = k.get("_pre")
                _cdiag("    t=%5.1fs conf=%.2f  APRES haut=%.2f bas=%.2f h=%.2f | AVANT %s  (%s)" % (k["t"], k["conf"], _wtop, _wbot, _hgt, (str(_pr) if _pr else "-"), k["kind"]))
            for k in out: k.pop("_pre", None)   # ne pas renvoyer ce champ de debug au client
        except Exception as _de:
            _cdiag("  diag terrain err: %s" % _de)
            for k in out: k.pop("_pre", None)
        _metrics = {}
        try:
            _oks = sum(1 for k in out if k.get("ok"))
            _ae = np.array(_anc_errs, dtype="float64") if _anc_errs else np.array([0.0])
            _sr = np.array(_snap_res, dtype="float64") if _snap_res else np.array([0.0])
            _px2cm = 10500.0 / max(1.0, Ws)  # 105 m de large ~ image 960px -> cm/px (approx)
            _metrics = {
                "t_total": round(_tSnap - _tA, 2), "t_motion": round(_tMot - _tA, 2),
                "t_anchors": round(_tAnc - _tMot, 2), "t_snap": round(_tSnap - _tAnc, 2),
                "n_anchors": len(anchors), "n_frames": len(out),
                "anc_err_moy": round(float(_ae.mean()), 2), "anc_err_max": round(float(_ae.max()), 2),
                "ligne_px_moy": round(float(_sr.mean()), 2), "ligne_px_max": round(float(_sr.max()), 2),
                "ligne_cm_moy": round(float(_sr.mean()) * _px2cm, 1),
                "bonnes": _oks, "pct_bonnes": round(100.0 * _oks / max(1, len(out))),
                "degen_rejetees": _degen, "ancres_rejetees_proportions": _prop_rej,
            }
            print("\n===== CALIBRAGE — TEMPS & QUALITÉ =====")
            print("  Clip analysé %.1fs (%d images suivies)" % (max(0.0, b - a), n))
            print("  TEMPS total %.1fs  =  caméra %.1fs  +  ancres PnL %.1fs (%d ancres, %.2fs/ancre)  +  recalage lignes %.1fs (%d images)"
                  % (_metrics["t_total"], _metrics["t_motion"], _metrics["t_anchors"], _metrics["n_anchors"],
                     _metrics["t_anchors"] / max(1, _metrics["n_anchors"]), _metrics["t_snap"], _metrics["n_frames"]))
            print("  QUALITÉ ancres (err. reprojection PnL) : moy %.2f px | max %.2f px" % (_metrics["anc_err_moy"], _metrics["anc_err_max"]))
            print("  QUALITÉ lignes (écart vert↔blanc)      : moy %.2f px (~%.1f cm) | max %.2f px  [échelle 960px]"
                  % (_metrics["ligne_px_moy"], _metrics["ligne_cm_moy"], _metrics["ligne_px_max"]))
            print("  Images ‘bonnes’ (conf≥0.7) : %d / %d (%d%%)" % (_metrics["bonnes"], _metrics["n_frames"], _metrics["pct_bonnes"]))
            print("  Images dégénérées rejetées (anti-écrasement) : %d" % _degen)
            print("  Ancres rejetées (proportions invraisemblables) : %d" % _prop_rej)
            print("========================================\n", flush=True)
        except Exception as _e:
            print("metrics err:", _e)
        _cp(pid, 100, "terminé")
        _result = {"keyframes": out, "n": len(out), "anchors": len(anchors), "cam": cam, "field_ranges": field_ranges, "metrics": _metrics}
        _result = _json_safe(_result)   # types natifs -> plus de 500 silencieuse à la sérialisation
        try:
            import json as _json
            _sz = len(_json.dumps(_result))
            _cdiag("==> RETOUR calibrate_smart : OK — %d keyframes envoyées au client (anchors=%d, cam=%d) — payload %d octets, JSON OK" % (len(out), len(anchors), len(cam), _sz))
        except Exception as _je:
            _cdiag("==> RETOUR calibrate_smart : PAYLOAD NON-JSON malgré nettoyage -> " + str(_je))
        return _result
    except Exception as e:
        import traceback
        _cdiag("==> RETOUR calibrate_smart : ERREUR -> " + str(e))
        _cdiag(traceback.format_exc())
        traceback.print_exc(); return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/ai_status")
def ai_status():
    info = {"file_exists": FIELD_KP_PATH.exists(), "path": str(FIELD_KP_PATH)}
    try:
        info["file_mb"] = round(os.path.getsize(FIELD_KP_PATH) / 1e6, 1) if FIELD_KP_PATH.exists() else 0
    except Exception:
        pass
    try:
        import gdown  # noqa
        info["gdown"] = True
    except Exception:
        info["gdown"] = False
    info["pnl_kp_mb"] = round(os.path.getsize(PNL_KP_W) / 1e6, 1) if PNL_KP_W.exists() else 0
    info["pnl_lines_mb"] = round(os.path.getsize(PNL_LINES_W) / 1e6, 1) if PNL_LINES_W.exists() else 0
    info["pnl_loaded"] = bool(_PNL)
    try:
        import cv2
        info["csrt"] = hasattr(cv2, "TrackerCSRT_create") or (hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"))
        info["nano"] = hasattr(cv2, "TrackerNano_create")
        info["cv2"] = cv2.__version__
    except Exception as e:
        info["csrt"] = False; info["cv2_err"] = str(e)
    try:
        m = get_field_model()
        info["loaded"] = m is not None
        if m is not None:
            ks = None
            for obj in (getattr(m, "model", None),):
                ks = getattr(obj, "kpt_shape", None) or ks
            try:
                ks = ks or m.model.model[-1].kpt_shape
            except Exception:
                pass
            info["kpt_shape"] = list(ks) if ks is not None else None
            info["task"] = getattr(m, "task", None)
    except Exception as e:
        info["loaded"] = False
        info["error"] = str(e)
    return info

@app.get("/vid/{vkey}")
def vid_get(vkey: str):
    """Sert la vidéo en cache (par vkey) — pour le moteur d'export rapide WebCodecs côté navigateur."""
    p = _resolve_vid(vkey)
    if not (p and os.path.exists(p)):
        return JSONResponse({"error": "absent"}, status_code=404)
    # NE BLOQUE JAMAIS : on sert le meilleur fichier DÉJÀ prêt (all-intra > _seekc.mp4 > source).
    # La construction du MP4 rapide se fait en arrière-plan via /vid_status?build=1.
    fast = FAST_INTRA.get(vkey)
    if isinstance(fast, str) and os.path.exists(fast):
        p = fast
    else:
        cand = _intra_existing(p)
        if cand:
            FAST_INTRA[vkey] = cand
            p = cand
        else:
            s = _fast_source_for(p)
            if s:
                p = s
    low = p.lower()
    mt = "video/mp4" if low.endswith(".mp4") else ("video/webm" if low.endswith(".webm") else "video/quicktime")
    return FileResponse(p, media_type=mt, headers={"Cache-Control": "no-store"})

@app.get("/vid_status/{vkey}")
def vid_status(vkey: str, build: int = 0):
    """État de la version MP4 rapide (all-intra). ?build=1 lance la construction en arrière-plan
    (ne bloque jamais l'export : la génération part tout de suite, ce sera rapide au prochain export)."""
    p = _resolve_vid(vkey)
    st = FAST_INTRA.get(vkey)
    if st == "building":
        return {"ready": False, "building": True}
    ready_path = st if (isinstance(st, str) and os.path.exists(st)) else None
    if not ready_path and p:
        cand = _intra_existing(p)
        if cand:
            FAST_INTRA[vkey] = cand
            ready_path = cand
    if ready_path:
        return {"ready": True, "building": False}
    if build and p and os.path.exists(p):
        FAST_INTRA[vkey] = "building"
        threading.Thread(target=_build_intra_bg, args=(vkey, p), daemon=True).start()
        return {"ready": False, "building": True}
    return {"ready": False, "building": False}

def _expected_size(vkey):
    """Taille attendue du fichier, encodée en suffixe du vkey ('..._2670259730')."""
    m = re.search(r"_(\d{4,})$", vkey or "")
    try:
        return int(m.group(1)) if m else 0
    except Exception:
        return 0

def _on_disk_original_size(vkey):
    """Taille du fichier D'ORIGINE (pas un dérivé _seek*) présent sur le disque, 0 si absent."""
    try:
        import glob
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", vkey or "")
        if not safe:
            return 0
        best = 0
        for c in glob.glob(os.path.join(str(VID), glob.escape(safe) + "*")):
            if not os.path.isfile(c):
                continue
            b = os.path.basename(c)
            if "_seekc" in b or "_seekI" in b or "_seek." in b or b.endswith(".part"):
                continue  # dérivés / uploads partiels : on ne les compte pas comme l'original
            best = max(best, os.path.getsize(c))
        return best
    except Exception:
        return 0

@app.get("/have/{vkey}")
def have(vkey: str):
    p = VKEYS.get(vkey)
    if not (p and os.path.exists(p) and _is_real_video(p)):
        # Après un REDÉMARRAGE du serveur, VKEYS (en mémoire) est vide : on retrouve la vidéo déjà
        # en cache sur le disque (on IGNORE les temporaires .part / _win_ pour ne jamais pointer dessus).
        try:
            import glob
            safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", vkey or "")
            if safe:
                cands = [c for c in glob.glob(os.path.join(str(VID), glob.escape(safe) + "*"))
                         if os.path.isfile(c) and _is_real_video(c)]
                seekc = [c for c in cands if c.endswith("_seekc.mp4") and _mp4_ok(c)]
                orig = [c for c in cands if not any(s in os.path.basename(c) for s in ("_seekc", "_seekI", "_seek."))]
                pick = seekc or orig or cands
                if pick:
                    VKEYS[vkey] = pick[0]; p = pick[0]
        except Exception:
            pass
    present = bool(p and os.path.exists(p))
    # « complete » = le fichier d'origine sur le disque a bien la taille attendue (upload non tronqué).
    exp = _expected_size(vkey)
    on_disk = _on_disk_original_size(vkey)
    if exp > 0:
        complete = present and on_disk >= int(exp * 0.995)
    else:
        complete = present  # pas de taille attendue connue -> on ne bloque pas
    return {"present": present, "complete": bool(complete), "size": on_disk, "expected": exp}

@app.get("/part_size/{vkey}")
def part_size(vkey: str):
    """Taille déjà reçue du gros fichier en cours d'envoi (fichier .part).
    Sert à REPRENDRE l'upload exactement où il s'est arrêté au lieu de tout renvoyer."""
    try:
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", vkey or "")
        if safe:
            part = VID / (safe + ".part")
            if part.exists():
                return {"size": int(part.stat().st_size)}
    except Exception:
        pass
    return {"size": 0}

@app.post("/put_chunk")
async def put_chunk(vkey: str = Form(...),
                    offset: int = Form(...),
                    total: int = Form(...),
                    name: str = Form(""),
                    last: int = Form(0),
                    file: UploadFile = File(...)):
    """Upload FIABLE par morceaux : écrit chaque bloc à son offset dans un .part, puis finalise
    en vérifiant la taille (anti-troncature du gros fichier du match)."""
    os.makedirs(VID, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", vkey or uuid.uuid4().hex)
    part = VID / (safe + ".part")
    mode = "r+b" if (int(offset) > 0 and part.exists()) else "wb"
    with open(part, mode) as f:
        f.seek(int(offset))
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    cur = part.stat().st_size
    if not int(last):
        return {"ok": True, "size": cur}
    # dernier morceau : on vérifie la taille avant de valider
    if int(total) > 0 and cur < int(total) * 0.995:
        return JSONResponse({"ok": False, "error": "incomplete", "size": cur, "expected": int(total)}, status_code=400)
    ext = os.path.splitext(name or "video.mp4")[1] or ".mp4"
    final = VID / (safe + ext)
    try:
        if final.exists():
            os.remove(final)
    except Exception:
        pass
    # on écarte aussi les anciens dérivés (seek/tl) devenus obsolètes
    for suf in ("_seekc.mp4", "_seek.mp4", "_seekI.mp4", "_seekc_seekI.mp4", "_tl.mp4"):
        try:
            d = VID / (safe + suf)
            if d.exists():
                os.remove(d)
        except Exception:
            pass
    os.replace(str(part), str(final))
    if vkey:
        VKEYS[vkey] = str(final)   # dispo tout de suite (le WebM complet est seekable en natif)
    # Ré-encodage seekable + all-intra EN ARRIÈRE-PLAN : ne bloque pas la fin de l'upload.
    def _bg_prepare(vk, fpath):
        try:
            sp = _ensure_seekable(fpath)
            if vk and sp and os.path.exists(sp):
                VKEYS[vk] = sp
            _build_intra_bg(vk, sp or fpath)
        except Exception as e:
            print("bg_prepare err:", e, flush=True)
    threading.Thread(target=_bg_prepare, args=(vkey, str(final)), daemon=True).start()
    return {"ok": True, "size": cur, "expected": int(total), "final": os.path.basename(str(final))}

@app.post("/upload")
async def upload(file: UploadFile = File(None),
                 vkey: str = Form(""),
                 model: str = Form("yolo11m.pt"),
                 conf: float = Form(0.3),
                 min_len: int = Form(8),
                 stride: int = Form(1),
                 imgsz: int = Form(640),
                 t0: float = Form(0.0),
                 t1: float = Form(0.0),
                 poly: str = Form(""),
                 maxp: int = Form(26),
                 cals: str = Form("")):
    # 1) Résoudre la vidéo : déjà en cache (vkey) ou bien envoyée maintenant.
    vpath = None
    if vkey and VKEYS.get(vkey) and os.path.exists(VKEYS[vkey]):
        vpath = VKEYS[vkey]
    if file is not None:
        ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", vkey or uuid.uuid4().hex)
        vp = VID / (safe + ext)
        with open(vp, "wb") as f:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        vpath = _ensure_seekable(str(vp))
        if vkey:
            VKEYS[vkey] = vpath
    if not vpath:
        return JSONResponse({"error": "video_absente", "need_file": True}, status_code=409)

    job_id = uuid.uuid4().hex[:10]
    d = WORK / job_id
    d.mkdir(parents=True, exist_ok=True)
    JOBS[job_id] = {"state": "queued", "progress": 0, "msg": "En file d'attente…",
                    "dir": d, "video": vpath, "name": (vkey or os.path.basename(vpath)),
                    "fps": 0, "w": 0, "h": 0, "frames": 0, "dur": 0, "cancel": False}
    pg = _parse_poly(poly); cl = _parse_cals(cals)
    threading.Thread(target=run_tracking,
                     args=(job_id, vpath, model, float(conf), int(min_len),
                           max(1, int(stride)), float(t0), float(t1), int(imgsz), pg, int(maxp), cl),
                     daemon=True).start()
    return {"job": job_id}

@app.get("/status/{job_id}")
def status(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        return JSONResponse({"error": "inconnu"}, status_code=404)
    return {k: j.get(k) for k in ("state", "progress", "msg", "fps", "w", "h", "frames", "dur", "name")}

@app.post("/cancel/{job_id}")
def cancel(job_id: str):
    if job_id in JOBS:
        JOBS[job_id]["cancel"] = True
    return {"ok": True}

@app.get("/tracks/{job_id}")
def tracks(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        return JSONResponse({"error": "inconnu"}, status_code=404)
    f = j["dir"] / "tracks.json"
    if not f.exists():
        return JSONResponse({"error": "pas encore prêt"}, status_code=409)
    return FileResponse(str(f), media_type="application/json", filename="tracks.json")

@app.get("/video/{job_id}")
def video(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        return PlainTextResponse("inconnu", status_code=404)
    return FileResponse(j["video"])

# ===== RELAIS DIRECT (LiveTag-like) : RTSP/RTMP/SRT/TCP/UDP -> HLS lisible par le navigateur =====
LIVE_RELAYS = {}

def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

@app.post("/live_relay")
def live_relay(url: str = Form(...)):
    """Convertit un flux réseau (rtsp/rtmp/srt/tcp/udp) en HLS local que le navigateur sait lire."""
    if not re.match(r"^(rtsp|rtmp|rtmps|srt|tcp|udp)://", url.strip(), re.I):
        return JSONResponse({"error": "URL non supportée (rtsp/rtmp/rtmps/srt/tcp/udp attendus)"}, status_code=400)
    rid = uuid.uuid4().hex[:8]
    d = WORK / ("relay_" + rid); d.mkdir(parents=True, exist_ok=True)
    out = d / "index.m3u8"
    ff = _ffmpeg_exe()
    base = [ff, "-loglevel", "warning", "-fflags", "nobuffer", "-flags", "low_delay"]
    if url.strip().lower().startswith("rtsp"):
        base += ["-rtsp_transport", "tcp"]
    hls = ["-f", "hls", "-hls_time", "1", "-hls_list_size", "6",
           "-hls_flags", "delete_segments+append_list", str(out)]
    # 1er essai : copie directe (caméras h264 = zéro latence CPU) ; repli : ré-encodage léger
    cmd_copy = base + ["-i", url, "-c:v", "copy", "-an"] + hls
    cmd_enc = base + ["-i", url, "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                      "-pix_fmt", "yuv420p", "-an"] + hls
    try:
        p = subprocess.Popen(cmd_copy, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        LIVE_RELAYS[rid] = {"p": p, "dir": d, "url": url, "cmd_enc": cmd_enc, "fallback": False}

        def _watch():
            import time
            time.sleep(6)
            info = LIVE_RELAYS.get(rid)
            if not info or info.get("fallback"):
                return
            if not out.exists() and info["p"].poll() is not None:
                # la copie directe a échoué (codec non h264) -> ré-encodage
                try:
                    info["p"].kill()
                except Exception:
                    pass
                try:
                    info["p"] = subprocess.Popen(cmd_enc, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    info["fallback"] = True
                except Exception:
                    pass
        threading.Thread(target=_watch, daemon=True).start()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return {"id": rid, "playlist": f"/relay/{rid}/index.m3u8"}

@app.get("/relay/{rid}/{fname}")
def relay_file(rid: str, fname: str):
    if not re.match(r"^[A-Za-z0-9_.-]+$", fname) or ".." in fname:
        return PlainTextResponse("nope", status_code=400)
    f = WORK / ("relay_" + rid) / fname
    if not f.exists():
        return PlainTextResponse("pas encore", status_code=404)
    mt = "application/vnd.apple.mpegurl" if fname.endswith(".m3u8") else "video/mp2t"
    return FileResponse(str(f), media_type=mt, headers={"Cache-Control": "no-store"})

@app.post("/live_relay_stop")
def live_relay_stop(id: str = Form(...)):
    info = LIVE_RELAYS.pop(id, None)
    if info:
        try:
            info["p"].kill()
        except Exception:
            pass
        try:
            import shutil
            shutil.rmtree(info["dir"], ignore_errors=True)
        except Exception:
            pass
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def index():
    # App d'analyse complète (même origine que le moteur => suivi IA intégré sans blocage)
    f = BASE / "app.html"
    html = f.read_text(encoding="utf-8") if f.exists() else (BASE / "ui.html").read_text(encoding="utf-8")
    # anti-cache : le navigateur recharge toujours la dernière version de l'app
    return HTMLResponse(html, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                                       "Pragma": "no-cache", "Expires": "0"})

@app.get("/studio", response_class=HTMLResponse)
def studio():
    # Outil de tracking autonome (glisser une vidéo, voir tous les joueurs)
    return (BASE / "ui.html").read_text(encoding="utf-8")

@app.on_event("startup")
def _warmup_models():
    """VITESSE : on charge (et on 'échauffe') les modèles au démarrage, en tâche de fond.
    Ainsi la 1re calibration et le 1er suivi ne subissent plus l'attente de chargement.
    En service auto (arrière-plan permanent), on saute ce préchauffage pour rester léger :
    les modèles se chargeront à la 1re utilisation du suivi/calibrage."""
    import os as _os
    # Par DÉFAUT on NE PRÉCHARGE PAS les modèles : ils occupent ~4 Go de RAM et, sur un Mac 8 Go,
    # saturent la mémoire -> swap -> TOUT le rendu du montage devient 2× plus lent. Les modèles se
    # chargent tout seuls à la 1re utilisation du Suivi IA / calibrage (léger délai la 1re fois, une seule fois).
    # Pour forcer le préchauffage (machines avec beaucoup de RAM) : lancer avec MATCHLYZ_WARMUP=1.
    if _os.environ.get("MATCHLYZ_WARMUP") != "1":
        return
    if _os.environ.get("MATCHLYZ_NO_WARMUP") == "1":
        return
    def _w():
        try:
            import numpy as _np
            dummy = _np.zeros((320, 320, 3), dtype="uint8")
            try:
                pm = get_player_model()
                if pm is not None:
                    pm.predict(dummy, verbose=False, **_pred_kw(pm, 640, pick_device()))
            except Exception as e:
                print("warmup joueur:", e)
            try:
                if get_reid() is not None:
                    reid_embed(dummy, [(160, 160, 40, 80)])
            except Exception as e:
                print("warmup reid:", e)
            try:
                fm = get_field_model()
                if fm is not None:
                    # INFÉRENCE de préchauffe : force la COMPILATION CoreML MAINTENANT (au démarrage,
                    # en tâche de fond) au lieu de la subir à la 1re calibration (= le blocage
                    # "Loading …mlpackage for CoreML inference…").
                    try:
                        fm.predict(dummy, conf=0.2, imgsz=getattr(fm, "_imgsz", 1280), verbose=False)
                    except Exception as _fe:
                        print("warmup terrain (inférence):", _fe)
                    print("Modèle terrain préchauffé (CoreML compilé).", flush=True)
            except Exception as e:
                print("warmup terrain:", e)
            try:
                get_pnl()
                # échauffe la 1re inférence PnL sur une VRAIE image (2 passes) : sinon le 1er clip calibré
                # après démarrage est "à froid" et de moins bonne qualité que les suivants.
                import glob as _g, cv2 as _cv2
                _wf = None
                try:
                    _vids = _g.glob(str(VID / "*.mp4")) + _g.glob(str(VID / "*.mov"))
                    if _vids:
                        _c = _cv2.VideoCapture(_vids[0]); _c.set(_cv2.CAP_PROP_POS_MSEC, 5000)
                        _okw, _wf = _c.read(); _c.release()
                except Exception:
                    _wf = None
                if _wf is None:
                    _wf = _np.zeros((720, 1280, 3), dtype="uint8")
                for _ in range(2):
                    try: pnl_field_pts(_wf)
                    except Exception: pass
            except Exception as e:
                print("warmup pnl:", e)
            print("Modèles préchargés — démarrage à chaud prêt.")
        except Exception as e:
            print("warmup global:", e)
    threading.Thread(target=_w, daemon=True).start()

if __name__ == "__main__":
    import uvicorn, sys
    sys.path.insert(0, str(BASE))
    print("\n  Foot Session  —  ouvre  http://localhost:8000  dans ton navigateur")
    print("  (app complète avec suivi IA intégré ; outil tracking seul sur /studio)")
    print("  Rechargement auto du code actif : les mises à jour s'appliquent sans relancer.\n")
    # PAS d'auto-reload : modele IA charge une fois par session.
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
