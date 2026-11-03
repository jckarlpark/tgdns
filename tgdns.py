"""
tgdns -- Taylor–Green 와류 3차원 의사 스펙트럴 직접수치모사(DNS) 풀이기.

논문 「Taylor–Green 와류의 직접수치모사(DNS)로 본 소산율과 최대 와도의 Reynolds 수 의존성」의
모든 계산이 이 모듈의 풀이기를 씁니다. 세 노트북(01_simulate, 02_analysis, 03_validation)이 공통으로 가져다 씁니다.

- 방법: 회전형 du/dt = P[u x w] + nu*lap(u), 주기 상자 [0, 2pi]^3, 2/3 비에일리어싱(|k_i| < N/3),
        적분 인자 RK4, 시간 간격 dt = min(dtmax, cfl*dx/u_max)
- 백엔드: CuPy가 있고 GPU가 보이면 GPU, 아니면 NumPy(+scipy.fft). set_backend("cpu")로 CPU를 강제할 수 있습니다.
- 출력 폴더: results/ (결과 pkl, 그림, 요약 txt)
"""
import os, time, pickle, zipfile
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

try:
    import scipy.fft as _sfft
    _rfftn = lambda a, axes: _sfft.rfftn(a, axes=axes, workers=-1)
    _irfftn = lambda a, s, axes: _sfft.irfftn(a, s=s, axes=axes, workers=-1)
except Exception:
    _rfftn = lambda a, axes: np.fft.rfftn(a, axes=axes)
    _irfftn = lambda a, s, axes: np.fft.irfftn(a, s=s, axes=axes)


class _CpuFFT:
    """CPU용 FFT 묶음 (3차원은 scipy.fft 다중 스레드, 나머지는 numpy)."""
    fftfreq = staticmethod(np.fft.fftfreq)
    rfftfreq = staticmethod(np.fft.rfftfreq)
    rfft2 = staticmethod(np.fft.rfft2)
    irfft2 = staticmethod(np.fft.irfft2)
    @staticmethod
    def rfftn(a, axes=None): return _rfftn(a, axes)
    @staticmethod
    def irfftn(a, s=None, axes=None): return _irfftn(a, s, axes)


xp, fft, GPU = np, _CpuFFT, False

def set_backend(name="auto"):
    """'auto'(GPU가 있으면 GPU), 'gpu', 'cpu'. 이미 만든 풀이기 객체에는 영향이 없으니 바꾼 뒤에 새로 만드세요."""
    global xp, fft, GPU
    use_gpu = False
    if name in ("auto", "gpu") and cp is not None:
        try:
            cp.zeros(1); use_gpu = True
        except Exception:
            use_gpu = False
    if name == "gpu" and not use_gpu:
        raise RuntimeError("GPU(cupy)를 쓸 수 없습니다.")
    xp, fft, GPU = (cp, cp.fft, True) if use_gpu else (np, _CpuFFT, False)
    print("GPU(cupy) 사용" if GPU else "CPU(numpy) 사용 -- 64^3 이하만 현실적입니다")
    return GPU

set_backend("auto")

def to_np(a):
    return cp.asnumpy(a) if GPU else np.asarray(a)

def free_gpu():
    if GPU:
        cp.get_default_memory_pool().free_all_blocks()

def ifrk4_step(uh, dt, L, nonlin):
    """적분 인자 RK4: 점성항은 정확히, 비선형항만 RK4."""
    E = xp.exp(L * dt / 2); E2 = E * E
    a = dt * nonlin(uh)
    b = dt * nonlin(E * (uh + a / 2))
    c = dt * nonlin(E * uh + b / 2)
    d = dt * nonlin(E2 * uh + E * c)
    return E2 * uh + (E2 * a + 2 * E * (b + c) + d) / 6

class Spectral3D:
    """주기 상자 [0,2pi]^3, du/dt = P[u x w] + nu*lap(u), 2/3 비에일리어싱(|k_i| < N/3)."""
    def __init__(self, N, nu):
        self.N, self.nu = N, nu
        k = fft.fftfreq(N, 1.0 / N); kz = fft.rfftfreq(N, 1.0 / N)
        self.kx, self.ky, self.kz = xp.meshgrid(k, k, kz, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2i = xp.where(self.k2 > 0, 1.0 / xp.where(self.k2 > 0, self.k2, 1.0), 0.0)
        kmax = N / 3.0
        self.mask = ((xp.abs(self.kx) < kmax) & (xp.abs(self.ky) < kmax) & (self.kz < kmax)).astype(float)
        self.L = -nu * self.k2
        w = xp.full(self.k2.shape, 2.0); w[..., 0] = 1.0
        if N % 2 == 0: w[..., -1] = 1.0
        self.w = w
        self.shell = np.rint(np.sqrt(to_np(self.k2))).astype(int).ravel()
        self.nshell = N // 2 + 1
    def phys(self, fh):
        return fft.irfftn(fh, s=(self.N,) * 3, axes=(1, 2, 3))
    def vorticity_hat(self, uh):
        kx, ky, kz = self.kx, self.ky, self.kz
        return 1j * xp.stack([ky * uh[2] - kz * uh[1], kz * uh[0] - kx * uh[2], kx * uh[1] - ky * uh[0]])
    def nonlin(self, uh):
        u = self.phys(uh); w = self.phys(self.vorticity_hat(uh))
        cr = xp.stack([u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]])
        nh = fft.rfftn(cr, axes=(1, 2, 3)) * self.mask
        kdn = self.kx * nh[0] + self.ky * nh[1] + self.kz * nh[2]
        return nh - xp.stack([self.kx, self.ky, self.kz]) * (kdn * self.k2i)
    def taylor_green(self):
        N = self.N; x = xp.arange(N) * (2 * np.pi / N)
        X, Y, Z = xp.meshgrid(x, x, x, indexing="ij")
        u = xp.stack([xp.sin(X) * xp.cos(Y) * xp.cos(Z), -xp.cos(X) * xp.sin(Y) * xp.cos(Z), xp.zeros_like(X)])
        return fft.rfftn(u, axes=(1, 2, 3)) * self.mask
    def diagnostics(self, uh):
        N6 = float(self.N) ** 6
        mode = (xp.abs(uh) ** 2).sum(axis=0) * self.w
        E = float(0.5 * mode.sum() / N6); Om = float(0.5 * (self.k2 * mode).sum() / N6)
        om = self.phys(self.vorticity_hat(uh)); u = self.phys(uh)
        wmax = float(xp.sqrt((om**2).sum(axis=0)).max()); umax = float(xp.sqrt((u**2).sum(axis=0)).max())
        return E, 2 * self.nu * Om, wmax, Om, umax
    def spectrum(self, uh):
        N6 = float(self.N) ** 6
        mode = 0.5 * (xp.abs(uh) ** 2).sum(axis=0) * self.w / N6
        Ek = np.bincount(self.shell, weights=to_np(mode).ravel(), minlength=self.nshell + 2)
        return Ek[: self.nshell]

def run_3d(N, Re, dtmax=0.01, T=10.0, t_rec=0.1, cfl=0.5, spec_times=(0, 2, 4, 6, 8, 8.5, 9, 9.5, 10), verbose=True):
    nu = 1.0 / Re; S = Spectral3D(N, nu); uh = S.taylor_green(); dx = 2 * np.pi / N
    t = 0.0; rec = []; spectra = {}
    E, eps, wmax, Om, umax = S.diagnostics(uh); rec.append((t, E, eps, wmax, Om, umax)); spectra[0.0] = S.spectrum(uh)
    nxt = t_rec; left = [s for s in spec_times if s > 0]
    dt = min(dtmax, cfl * dx / umax); t0 = time.time(); nstep = 0; nprint = 0
    while t < T - 1e-12:
        h = min(dt, nxt - t)
        uh = ifrk4_step(uh, h, S.L, S.nonlin); t += h; nstep += 1
        if t >= nxt - 1e-12:
            E, eps, wmax, Om, umax = S.diagnostics(uh)
            if not np.isfinite(E):
                print(f"  !! 발산 N={N} Re={Re} t={t:.2f}"); break
            rec.append((t, E, eps, wmax, Om, umax))
            dt = min(dtmax, cfl * dx / max(umax, 1e-6)); nxt += t_rec
            if left and t >= left[0] - 1e-9:
                spectra[round(left[0], 3)] = S.spectrum(uh); left.pop(0)
            if verbose and int(t) > nprint and int(t) % 2 == 0:
                nprint = int(t); print(f"    t={t:5.2f}  eps={eps:.5f}  wmax={wmax:7.2f}  ({time.time()-t0:.0f}s)")
    a = np.array(rec)
    out = {"N": N, "Re": Re, "dtmax": dtmax, "t": a[:, 0], "E": a[:, 1], "eps": a[:, 2], "wmax": a[:, 3],
           "Om": a[:, 4], "umax": a[:, 5], "spectra": spectra, "walltime": time.time() - t0, "nstep": nstep}
    free_gpu()
    return out

class Spectral2D:
    def __init__(self, N, nu):
        self.N, self.nu = N, nu
        k = fft.fftfreq(N, 1.0 / N); kz = fft.rfftfreq(N, 1.0 / N)
        self.kx, self.ky = xp.meshgrid(k, kz, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2
        self.k2i = xp.where(self.k2 > 0, 1.0 / xp.where(self.k2 > 0, self.k2, 1.0), 0.0)
        kmax = N / 3.0
        self.mask = ((xp.abs(self.kx) < kmax) & (self.ky < kmax)).astype(float)
        self.L = -nu * self.k2
        w = xp.full(self.k2.shape, 2.0); w[:, 0] = 1.0
        if N % 2 == 0: w[:, -1] = 1.0
        self.w = w
    def nonlin(self, wh):
        psih = wh * self.k2i; s = (self.N,) * 2
        u = fft.irfft2(1j * self.ky * psih, s=s); v = fft.irfft2(-1j * self.kx * psih, s=s)
        wx = fft.irfft2(1j * self.kx * wh, s=s); wy = fft.irfft2(1j * self.ky * wh, s=s)
        return -fft.rfft2(u * wx + v * wy) * self.mask
    def enstrophy(self, wh):
        return float(0.5 * (self.w * xp.abs(wh) ** 2).sum() / self.N**4)
    def energy(self, wh):
        return float(0.5 * (self.w * self.k2i * xp.abs(wh) ** 2).sum() / self.N**4)

def run_2d(N, Re, E0=0.125, T=10.0, t_rec=0.1, cfl=0.5, seed=0):
    S = Spectral2D(N, 1.0 / Re); rng = np.random.default_rng(seed)
    kx = to_np(S.kx); ky = to_np(S.ky); mag = np.sqrt(kx**2 + ky**2)
    ph = np.exp(2j * np.pi * rng.random(kx.shape))
    wh = xp.asarray(((mag > 0) & (mag <= 4)) * ph) * S.mask
    wh = fft.rfft2(fft.irfft2(wh, s=(N, N))) * S.mask
    wh = wh * np.sqrt(E0 / S.energy(wh))
    t = 0.0; out = [(0.0, S.enstrophy(wh))]; nxt = t_rec; dx = 2 * np.pi / N; dt = 0.01
    while t < T - 1e-12:
        h = min(dt, nxt - t); wh = ifrk4_step(wh, h, S.L, S.nonlin); t += h
        if t >= nxt - 1e-12:
            out.append((t, S.enstrophy(wh))); psih = wh * S.k2i; s = (N, N)
            um = float(xp.sqrt(fft.irfft2(1j * S.ky * psih, s=s) ** 2 + fft.irfft2(-1j * S.kx * psih, s=s) ** 2).max())
            dt = min(0.01, cfl * dx / max(um, 1e-6)); nxt += t_rec
    return np.array(out)

def at(r, key, t_target):
    i = int(np.argmin(np.abs(r["t"] - t_target))); return r["t"][i], r[key][i]

def trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))

print("정의 완료")

# ---------------------------------------------------------------------------
# 검증용 추가 기능 (03_validation에서 사용): 헬리시티, L3 노름, 매 단계 에너지 수지
# ---------------------------------------------------------------------------
def _enstrophy_energy(S, uh):
    N6 = float(S.N) ** 6; mode = (xp.abs(uh) ** 2).sum(axis=0) * S.w
    return float(0.5 * (S.k2 * mode).sum() / N6), float(0.5 * mode.sum() / N6)

def _diagnostics_ext(S, uh):
    Om, E = _enstrophy_energy(S, uh)
    om = S.phys(S.vorticity_hat(uh)); u = S.phys(uh)
    umag = xp.sqrt((u ** 2).sum(axis=0))
    return dict(E=E, eps=2 * S.nu * Om, wmax=float(xp.sqrt((om ** 2).sum(axis=0)).max()),
                umax=float(umag.max()), hel=float((u * om).sum(axis=0).mean()),
                L3=float(xp.mean(umag ** 3)) ** (1 / 3))

def run_3d_checks(N, Re, dtmax=0.01, T=10.0, t_rec=0.1, cfl=0.5, perstep=False):
    """run_3d와 같은 시간 간격 규칙. perstep=True이면 매 단계 eps를 사다리꼴로 적분(에너지 수지 점검)."""
    nu = 1.0 / Re; S = Spectral3D(N, nu); uh = S.taylor_green(); dx = 2 * np.pi / N
    t = 0.0; d = _diagnostics_ext(S, uh); rec = [dict(t=t, **d)]
    dt = min(dtmax, cfl * dx / d["umax"]); nxt = t_rec; nstep = 0
    I = 0.0; eprev = d["eps"]
    while t < T - 1e-12:
        h = min(dt, nxt - t); uh = ifrk4_step(uh, h, S.L, S.nonlin); t += h; nstep += 1
        if perstep:
            ecur = 2 * nu * _enstrophy_energy(S, uh)[0]; I += 0.5 * (eprev + ecur) * h; eprev = ecur
        if t >= nxt - 1e-12:
            d = _diagnostics_ext(S, uh); rec.append(dict(t=t, **d))
            dt = min(dtmax, cfl * dx / max(d["umax"], 1e-6)); nxt += t_rec
    out = {k: np.array([r[k] for r in rec]) for k in rec[0]}
    out.update(N=N, Re=Re, dtmax=dtmax, nstep=nstep, I_perstep=I)
    free_gpu()
    return out


# ---------------------------------------------------------------------------
# 해상도 지표와 입출력 도우미
# ---------------------------------------------------------------------------
def resolution_index(r):
    """k_max*eta (k_max = N/3, eta는 소산율 피크에서의 값)."""
    ip = int(np.argmax(r["eps"])); nu = 1.0 / r["Re"]
    return r["N"] / 3 * (nu ** 3 / r["eps"][ip]) ** 0.25

def verdict(k):
    return "충분" if k >= 1.1 else ("경계" if k >= 0.8 else "부족")

RESULTS_DIR = "results"
DATA_DIR = "data"

def rpath(name):
    """results/ 안의 경로 (폴더가 없으면 만듭니다)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)

def load_pickle(name, default=None):
    """results/<name>을 읽고, 없으면 현재 폴더의 <name>을 읽고, 그것도 없으면 default."""
    for p in (os.path.join(RESULTS_DIR, name), name):
        if os.path.exists(p):
            return pickle.load(open(p, "rb"))
    return default

def save_pickle(name, obj):
    pickle.dump(obj, open(rpath(name), "wb"))

def zip_results(zipname="tg_outputs.zip"):
    """results/ 전체를 압축하고, 코랩이면 내려받습니다."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(RESULTS_DIR) if not f.endswith(".gdiag"))
    with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(os.path.join(RESULTS_DIR, f), arcname=f)
    print("압축한 파일:", files)
    try:
        from google.colab import files as _cf
        _cf.download(zipname)
    except Exception as e:
        print("(자동 다운로드는 코랩에서만 동작합니다)", type(e).__name__)
