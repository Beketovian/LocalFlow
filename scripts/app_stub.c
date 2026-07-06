/* LocalFlow.app main executable: runs the daemon via libpython, in-process.
 *
 * Why a compiled stub instead of a shell launcher: macOS TCC (Accessibility,
 * Input Monitoring, Microphone) records the identity of the app's main
 * executable. A launcher that exec()s another binary leaves the running
 * process with a different identity than the one the user granted, so the
 * permission toggle shows ON while AXIsProcessTrusted() stays false. With
 * this stub the granted executable IS the running process - one identity.
 *
 * Baked-in paths (libpython, PYTHONPATH) are injected by scripts/build_app.sh
 * via -D defines at compile time.
 */

#include <dlfcn.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef LIBPYTHON_PATH
#error "compile with -DLIBPYTHON_PATH=..."
#endif
#ifndef LOCALFLOW_PYTHONPATH
#error "compile with -DLOCALFLOW_PYTHONPATH=..."
#endif

typedef int (*py_bytes_main_t)(int, char **);

int main(int argc, char *argv[]) {
    (void)argc;

    /* Environment the daemon expects (mirrors the old shell launcher). */
    setenv("LOCALFLOW_APP", "1", 1);
    setenv("PYTHONUNBUFFERED", "1", 1);
    setenv("PYTHONPATH", LOCALFLOW_PYTHONPATH, 1);

    /* Send stdout/stderr to the log file - there is no terminal. */
    const char *home = getenv("HOME");
    if (home != NULL) {
        char log_path[PATH_MAX];
        snprintf(log_path, sizeof log_path, "%s/Library/Logs/LocalFlow.log", home);
        FILE *log = fopen(log_path, "a");
        if (log != NULL) {
            fprintf(log, "=== LocalFlow.app launched ===\n");
            fclose(log);
            freopen(log_path, "a", stdout);
            freopen(log_path, "a", stderr);
            setvbuf(stdout, NULL, _IONBF, 0);
            setvbuf(stderr, NULL, _IONBF, 0);
        }
    }

    void *libpython = dlopen(LIBPYTHON_PATH, RTLD_NOW | RTLD_GLOBAL);
    if (libpython == NULL) {
        fprintf(stderr, "LocalFlow: cannot load %s: %s\n", LIBPYTHON_PATH, dlerror());
        return 1;
    }
    py_bytes_main_t py_bytes_main =
        (py_bytes_main_t)dlsym(libpython, "Py_BytesMain");
    if (py_bytes_main == NULL) {
        fprintf(stderr, "LocalFlow: Py_BytesMain not found: %s\n", dlerror());
        return 1;
    }

    char *py_argv[] = {argv[0], "-m", "localflow.cli", "run", NULL};
    return py_bytes_main(4, py_argv);
}
