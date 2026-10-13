# CUDA MatrixLab

CUDA MatrixLab is a learning and benchmarking project with four parts:

- `cuda/`: C++ and CUDA programs. A CPU baseline, a naive CUDA matrix multiply, a shared-memory tiled CUDA matrix multiply, and a CUDA vector add. Each one verifies its own result and prints one JSON line.
- `colab/CUDA_MatrixLab.ipynb`: a Google Colab notebook that compiles and runs those programs on a real NVIDIA GPU and exports the measurements to JSON.
- `backend/`: a FastAPI service. It detects whether CUDA can run locally, compiles and runs the local CPU baseline, validates result files, and stores them.
- `frontend/`: a React, TypeScript, Vite and Tailwind interface. It shows the environment, imports Colab JSON, runs local benchmarks, and draws the results.

**What this project does and does not claim.** Every timing shown in the UI comes from a program run that is recorded in a results JSON file. Nothing is hard-coded or estimated. The UI never reports CUDA results from a machine without working CUDA. If your computer has no NVIDIA GPU, run the CUDA programs in Colab and import the JSON file.

---

## 1. Software to install first (Windows)

| Software | Needed for | Download |
|---|---|---|
| Visual Studio Code | Editing and running the project | https://code.visualstudio.com |
| Node.js LTS (20 or 22) | Frontend (`npm`) | https://nodejs.org |
| Python 3.11 or newer (3.12 recommended) | Backend | https://www.python.org/downloads/ (tick "Add python.exe to PATH") |
| MinGW-w64 g++ (optional) | Running the CPU baseline on your laptop | https://www.mingw-w64.org or `winget install MSYS2.MSYS2` (then install `mingw-w64-ucrt-x86_64-gcc`) |
| NVIDIA driver + CUDA Toolkit (optional) | Running CUDA on your laptop, only if you have an NVIDIA GPU | https://developer.nvidia.com/cuda-downloads |
| Google account | Running CUDA on a free Colab GPU | https://colab.research.google.com |

Check the tools from PowerShell:

```powershell
node -v
npm -v
python --version
g++ --version      # optional; needed only for local CPU runs
nvidia-smi         # optional; prints an error if you have no NVIDIA GPU
```

Local CUDA is available only if `nvidia-smi` works **and** `nvcc --version` works. Otherwise use Colab.

---

## 2. Open the project

Unzip the archive. In PowerShell:

```powershell
Expand-Archive -Path "$HOME\Downloads\CUDA-MatrixLab.zip" -DestinationPath "$HOME\Projects" -Force
cd "$HOME\Projects\CUDA-MatrixLab"
code .
```

In VS Code, open two integrated terminals (**Terminal → New Terminal**, or the `+` button). Both start in the project folder.

- **Terminal A** runs the frontend.
- **Terminal B** runs the backend.

If PowerShell refuses to run scripts that activate a virtual environment, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## 3. Frontend (Terminal A)

```powershell
cd "$HOME\Projects\CUDA-MatrixLab\frontend"
npm install
npm run build      # type-checks and builds; should end with "built in ..."
npm run dev        # keep this running
```

Open **http://localhost:5173** in your browser.

Shortcut: `..\scripts\start-frontend.ps1` does the install (only if needed) and starts the dev server.

---

## 4. Backend (Terminal B)

```powershell
cd "$HOME\Projects\CUDA-MatrixLab\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q                                  # runs the backend tests
uvicorn app.main:app --reload --port 8000
```

Keep this terminal open. The API is at **http://127.0.0.1:8000**. Interactive API docs are at **http://127.0.0.1:8000/docs**.

Shortcut: `..\scripts\start-backend.ps1` creates the virtual environment, installs dependencies, runs the tests, and starts the server.

To stop either server, press `Ctrl+C` in its terminal. To reactivate the virtual environment later, run `.\.venv\Scripts\Activate.ps1` from `backend\`.

**Which URL to open:** **http://localhost:5173** for the UI. The UI needs the backend running too. If the UI shows "Cannot reach the backend", Terminal B is not running.

---

## 5. Compile and run the C++ and CUDA programs

### 5a. Windows PowerShell: CPU baseline (no GPU needed, needs g++)

From the project root:

```powershell
New-Item -ItemType Directory -Force -Path build | Out-Null
g++ -O3 -std=c++17 -I cuda cuda\matrix_cpu.cpp -o build\matrix_cpu.exe
.\build\matrix_cpu.exe 512 3
```

The program prints a human-readable log to the terminal's error stream and one JSON line on standard output. To save the JSON:

```powershell
.\build\matrix_cpu.exe 512 3 > build\cpu_512.json
```

Or use the helper: `.\scripts\build-cpu.ps1 -N 512 -Reps 3`.

### 5b. Windows PowerShell: CUDA programs (only with a local NVIDIA GPU)

You need the NVIDIA driver, the CUDA Toolkit (which provides `nvcc`), and Visual Studio Build Tools with the "Desktop development with C++" workload. First find your GPU's compute capability on https://developer.nvidia.com/cuda-gpus (a T4 is 7.5, so use `75`). If your toolkit is 11.6 or newer, you can use `-arch=native` instead.

```powershell
New-Item -ItemType Directory -Force -Path build | Out-Null
nvcc -O3 -std=c++17 -arch=sm_75 -I cuda cuda\vector_add.cu -o build\vector_add.exe
nvcc -O3 -std=c++17 -arch=sm_75 -I cuda cuda\matrix_cuda_naive.cu -o build\matrix_cuda_naive.exe
nvcc -O3 -std=c++17 -arch=sm_75 -I cuda cuda\matrix_cuda_shared.cu -o build\matrix_cuda_shared.exe

.\build\vector_add.exe 67108864 20
.\build\matrix_cuda_naive.exe 2048 10
.\build\matrix_cuda_shared.exe 2048 10
```

Arguments: `<n> [reps]`. For `vector_add`, `n` is the number of elements. For the matrix programs, `n` is the side length of the square matrix.

The backend does this for you when local CUDA is available: choose a program in the **Run on this machine** panel.

### 5c. Google Colab (Linux, real NVIDIA GPU)

Colab commands are shown in the notebook. In a Colab cell, the equivalent commands are:

```bash
nvidia-smi
nvcc --version
cd /content && unzip -q CUDA-MatrixLab.zip
cd /content/CUDA-MatrixLab && mkdir -p build
g++ -O3 -std=c++17 -I cuda cuda/matrix_cpu.cpp -o build/matrix_cpu
nvcc -O3 -std=c++17 -arch=sm_75 -I cuda cuda/matrix_cuda_naive.cu -o build/matrix_cuda_naive
nvcc -O3 -std=c++17 -arch=sm_75 -I cuda cuda/matrix_cuda_shared.cu -o build/matrix_cuda_shared
nvcc -O3 -std=c++17 -arch=sm_75 -I cuda cuda/vector_add.cu -o build/vector_add
./build/vector_add 67108864 20
./build/matrix_cuda_shared 2048 10
```

Use `-arch=sm_75` for a T4. Colab may assign a different GPU, so the notebook detects the compute capability and uses it automatically.

---

## 6. Run the Colab notebook

1. Open **https://colab.research.google.com**, then **File → Upload notebook**, and choose `colab/CUDA_MatrixLab.ipynb`.
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU** → Save.
3. Run the first cell. It must print a GPU table. If it does not, change the runtime type and restart.
4. Run the **upload** cell and choose `CUDA-MatrixLab.zip` from your computer.
5. **Runtime → Run all**. The benchmark cell takes several minutes. Every line printed is a real measurement; the cell stops if any result fails verification.
6. The last cell downloads **`cuda_matrixlab_results.json`**. Keep this file.

---

## 7. Import Colab results into the UI

1. Make sure both servers are running (Section 3 and Section 4).
2. Open **http://localhost:5173**.
3. In the left column, under **Import Colab results**, click **Choose File** and select `cuda_matrixlab_results.json`.
4. The backend validates the file. On success, the **Results** panel shows the GPU, CUDA version, a verification badge, a table of CPU vs naive vs shared timings and speedups, a log-scale chart, and the vector-add bandwidth table.
5. You can import several files. Use the dropdown at the top of the Results panel to switch between them.

If validation fails, the message shows the field that does not match the schema. Do not edit the file by hand; re-export it from Colab.

---

## 8. Project layout

```
CUDA-MatrixLab/
├── README.md
├── cuda/
│   ├── common.h                  shared helpers: RNG, median, verification, JSON output
│   ├── common_cuda.h             CUDA_CHECK macro
│   ├── matrix_cpu.cpp            sequential CPU matrix multiply (i-k-j loop order)
│   ├── matrix_cuda_naive.cu      one thread per output element, global memory only
│   ├── matrix_cuda_shared.cu     32x32 shared-memory tiles
│   └── vector_add.cu             c = a + b
├── colab/
│   └── CUDA_MatrixLab.ipynb      runs everything on a Colab GPU and exports JSON
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py               FastAPI routes
│   │   ├── cuda_detect.py        local CUDA / g++ detection
│   │   ├── benchmark_runner.py   compiles and runs the programs
│   │   ├── results_store.py      saves results under backend/data/results/
│   │   ├── schemas.py            result JSON schema (pydantic)
│   │   └── config.py
│   └── tests/test_api.py
├── frontend/
│   ├── package.json, vite.config.ts, tailwind.config.js, tsconfig.json
│   └── src/                      App.tsx, components/, lib/, api.ts, types.ts
└── scripts/                      PowerShell helpers
```

### Results JSON format

```json
{
  "schema": "cuda-matrixlab-results/v1",
  "generated_at": "2026-...T...+00:00",
  "source": "google-colab",
  "environment": { "gpu_name": "...", "cuda_version": "...", "cpu_model": "..." },
  "runs": [
    {
      "kernel": "cuda_shared_matmul",
      "n": 2048, "reps": 10,
      "device": "...", "compute_capability": "7.5",
      "time_scope": "kernel_only_cuda_events",
      "times_ms": [ ... ], "median_ms": 0.0, "min_ms": 0.0,
      "metric": { "type": "gflops", "value": 0.0 },
      "verified": true, "max_abs_error": 0.0
    }
  ]
}
```

### Method notes

- **CUDA timing** uses `cudaEvent` pairs around each kernel launch, after one warm-up launch. Host-to-device copies are excluded, so the numbers compare kernel execution only.
- **CPU timing** uses a steady clock around the multiplication, after one warm-up run. It excludes allocation and initialisation.
- **Verification** recomputes 64 randomly chosen entries of C in double precision on the CPU. An entry passes if its error is within a relative 1e-3. Vector addition is checked element by element.
- **Throughput:** matrix multiply uses 2·n³ floating-point operations; vector add uses 3·n·4 bytes moved.
- Shared memory tiles are 32×32. A block has 1024 threads, the maximum on most NVIDIA GPUs.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot reach the backend` in the UI | Start Terminal B (`uvicorn app.main:app --reload --port 8000`). |
| `running scripts is disabled on this system` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then reopen the terminal. |
| `g++ was not found` | Install MinGW-w64/MSYS2 g++ and reopen VS Code, or run the CPU part in Colab. |
| `Local CUDA unavailable` | Expected on laptops without an NVIDIA GPU. Use Colab. |
| Colab cell says no GPU | Runtime → Change runtime type → T4 GPU, then restart the session. |
| Import says schema mismatch | Re-download the JSON from the notebook; do not edit it. |
| `npm install` fails behind a proxy | Check your network or proxy settings for the npm registry. |

---

## 10. Limitations

- Local CUDA execution is supported only on Windows or Linux machines with an NVIDIA GPU, driver and CUDA Toolkit. Most laptops without an NVIDIA GPU cannot run it.
- The backend's local CPU runs require g++ on PATH. The `cl` (MSVC) compiler is not used by the backend.
- Colab GPUs differ between sessions. Results record the GPU name and compute capability from the session that produced them.
- Benchmarks are single-run measurements on shared hardware. They vary with thermal state, background load and clock scaling. Repeat the notebook for stable numbers.
- The CPU baseline is single-threaded and not vectorised by hand. Compiler auto-vectorisation at `-O3` may still apply. Comparisons are against this baseline, not against optimised libraries such as cuBLAS or OpenBLAS.
- Verification is statistical (sampled entries), not a full recomputation, to keep large sizes practical.
