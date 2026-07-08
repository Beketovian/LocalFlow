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
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
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

/* Standalone bundles bake "@RESOURCES@/..." into the -D defines below
 * (PYTHONPATH, libpython, PYTHONHOME); expand it to
 * <bundle>/Contents/Resources at runtime so the app keeps working wherever
 * it is moved. */
static void resources_dir(char *out, size_t out_size) {
    out[0] = '\0';
    char exe[PATH_MAX];
    uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) == 0) {
        char real[PATH_MAX];
        if (realpath(exe, real) != NULL) {
            /* .../LocalFlow.app/Contents/MacOS/LocalFlow -> .../Contents */
            char *contents = dirname(dirname(real));
            snprintf(out, out_size, "%s/Resources", contents);
        }
    }
}

static void expand_template(const char *template, char *out, size_t out_size) {
    if (strstr(template, "@RESOURCES@") == NULL) {
        snprintf(out, out_size, "%s", template);
        return;
    }
    char resources[PATH_MAX];
    resources_dir(resources, sizeof resources);
    size_t pos = 0;
    for (const char *p = template; *p != '\0' && pos + 1 < out_size;) {
        if (strncmp(p, "@RESOURCES@", 11) == 0) {
            pos += (size_t)snprintf(out + pos, out_size - pos, "%s", resources);
            p += 11;
        } else {
            out[pos++] = *p++;
        }
    }
    out[pos < out_size ? pos : out_size - 1] = '\0';
}

int main(int argc, char *argv[]) {
    (void)argc;

    /* Environment the daemon expects (mirrors the old shell launcher). */
    char pythonpath[4 * PATH_MAX];
    expand_template(LOCALFLOW_PYTHONPATH, pythonpath, sizeof pythonpath);
    setenv("LOCALFLOW_APP", "1", 1);
    setenv("PYTHONUNBUFFERED", "1", 1);
    setenv("PYTHONPATH", pythonpath, 1);

    /* Where read-only bundled assets live (e.g. Resources/models with the
     * Whisper weights standalone builds ship). */
    char resources[PATH_MAX];
    resources_dir(resources, sizeof resources);
    if (resources[0] != '\0')
        setenv("LOCALFLOW_RESOURCES", resources, 1);

#ifdef LOCALFLOW_PYTHONHOME
    /* Standalone bundles carry their own Python runtime: without
     * PYTHONHOME, libpython would look for the stdlib next to this
     * executable (argv[0]) and fail to start. */
    char pythonhome[PATH_MAX];
    expand_template(LOCALFLOW_PYTHONHOME, pythonhome, sizeof pythonhome);
    setenv("PYTHONHOME", pythonhome, 1);
#endif

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

    char libpython_path[PATH_MAX];
    expand_template(LIBPYTHON_PATH, libpython_path, sizeof libpython_path);
    void *libpython = dlopen(libpython_path, RTLD_NOW | RTLD_GLOBAL);
    if (libpython == NULL) {
        fprintf(stderr, "LocalFlow: cannot load %s: %s\n", libpython_path, dlerror());
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
