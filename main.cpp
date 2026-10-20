// s3_viewer.cpp
// First-person view inside the 3-sphere S^3.
//
// Scene: the regular tetrahedral honeycomb {3,3,4} of S^3 (16 spherical rooms,
// the walls are the 4 great 2-spheres x_i = 0), 4 moons at the room corners
// (16-cell vertices e1..e4, deliberately no antipodal partners so each moon's
// far "wrap-around" image stays visible) plus one small bright "Polaris" moon.
//
// Rendering: every pixel marches along the great circle gamma(t) = cos(t)*p + sin(t)*v
// for t in (0, 2pi)  -- the full geodesic loop. This is exactly what a 3D creature
// living in S^3 sees: every object appears twice (near image and the "long way
// around" image behind it), and all rays reconverge at the antipodal point at t = pi.
//
// Controls:
//   W/S        move forward/backward along view
//   A/D        strafe left/right
//   R/F        move up/down
//   Q/E        roll counterclockwise/clockwise
//   mouse drag look around (yaw/pitch)
//   SHIFT      move faster
//   M          cycle wall opacity (translucent -> invisible -> nearly opaque)
//   T          reset pose
//   ESC        quit
//
// Build (mingw + vcpkg): see build.bat

#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

struct V4 { float x, y, z, w; };

static inline V4 vadd(V4 a, V4 b) { return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
static inline V4 vsub(V4 a, V4 b) { return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
static inline V4 vscale(V4 a, float s) { return { a.x * s, a.y * s, a.z * s, a.w * s }; }
static inline float vdot(V4 a, V4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
static inline V4 vnorm(V4 a) {
    float l = sqrtf(vdot(a, a));
    return l > 1e-12f ? vscale(a, 1.0f / l) : a;
}

// camera frame: eye in S^3, fwd/right/up orthonormal tangent frame at eye
static V4 eye, fwd, rgt, up;
static float pitchTotal = 0.0f;

static void resetPose() {
    eye = vnorm(V4{ 1.0f, 0.55f, 0.42f, 0.31f });
    V4 f0 = V4{ -0.3f, 1.0f, 0.2f, 0.4f };
    V4 r0 = V4{ 0.3f, 0.2f, 1.0f, -0.4f };
    V4 u0 = V4{ 0.2f, -0.3f, 0.4f, 1.0f };
    fwd = vnorm(vsub(f0, vscale(eye, vdot(f0, eye))));
    rgt = vnorm(vsub(vsub(r0, vscale(eye, vdot(r0, eye))), vscale(fwd, vdot(r0, fwd))));
    up  = vnorm(vsub(vsub(vsub(u0, vscale(eye, vdot(u0, eye))), vscale(fwd, vdot(u0, fwd))), vscale(rgt, vdot(u0, rgt))));
    pitchTotal = 0.0f;
}

// rotations of the tangent frame
static void rotYaw(float a) {
    float c = cosf(a), s = sinf(a);
    V4 nf = vadd(vscale(fwd, c), vscale(rgt, s));
    V4 nr = vadd(vscale(rgt, c), vscale(fwd, -s));
    fwd = nf; rgt = nr;
}
static void rotPitch(float b) {
    float c = cosf(b), s = sinf(b);
    V4 nf = vadd(vscale(fwd, c), vscale(up, -s));
    V4 nu = vadd(vscale(up, c), vscale(fwd, s));
    fwd = nf; up = nu;
}
static void rotRoll(float a) {
    float c = cosf(a), s = sinf(a);
    V4 nr = vadd(vscale(rgt, c), vscale(up, s));
    V4 nu = vadd(vscale(up, c), vscale(rgt, -s));
    rgt = nr; up = nu;
}

// move the eye along the geodesic with initial tangent d by arclength s,
// and parallel-transport the tangent frame along it.
static void moveAlong(V4 d, float s) {
    V4 ne = vadd(vscale(eye, cosf(s)), vscale(d, sinf(s)));
    auto tp = [&](V4 b) {
        float al = vdot(b, d);
        V4 bp = vsub(b, vscale(d, al));
        return vadd(bp, vadd(vscale(d, cosf(s) * al), vscale(eye, -sinf(s) * al)));
    };
    V4 nf = tp(fwd), nr = tp(rgt), nu = tp(up);
    // re-orthonormalize (only fixes float drift)
    nf = vnorm(vsub(nf, vscale(ne, vdot(ne, nf))));
    nr = vnorm(vsub(vsub(nr, vscale(ne, vdot(ne, nr))), vscale(nf, vdot(nf, nr))));
    nu = vnorm(vsub(vsub(vsub(nu, vscale(ne, vdot(ne, nu))), vscale(nf, vdot(nf, nu))), vscale(nr, vdot(nr, nu))));
    eye = ne; fwd = nf; rgt = nr; up = nu;
}

static const char* vertSrc = R"GLSL(
#version 330 core
out vec2 vUV;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

static const char* fragSrc = R"GLSL(
#version 330 core
in vec2 vUV;
out vec4 outColor;

uniform vec4  uEye, uF, uR, uU, uSun;
uniform float uAspect, uTanF, uWallAlpha;

const float PI    = 3.141592653589793;
const float TWOPI = 6.283185307179586;
const float EPS   = 1e-4;

struct Moon { vec4 c; vec3 col; float r; };

// 4 moons at room corners e1..e4 (deliberately no antipodal partners, so each
// moon's far "wrap-around" image stays visible) + small bright "Polaris" moon
const Moon moons[5] = Moon[](
    Moon(vec4( 1.0, 0.0, 0.0, 0.0), vec3(0.85, 0.30, 0.28), 0.24),
    Moon(vec4( 0.0, 1.0, 0.0, 0.0), vec3(0.32, 0.72, 0.36), 0.24),
    Moon(vec4( 0.0, 0.0, 1.0, 0.0), vec3(0.62, 0.45, 0.85), 0.24),
    Moon(vec4( 0.0, 0.0, 0.0, 1.0), vec3(0.45, 0.80, 0.80), 0.24),
    Moon(vec4(-0.5473, 0.7165, 0.3980, 0.1692), vec3(1.00, 0.98, 0.92), 0.07)
);

const vec3 wallCol[4] = vec3[](
    vec3(0.72, 0.35, 0.30),
    vec3(0.30, 0.56, 0.34),
    vec3(0.30, 0.42, 0.72),
    vec3(0.70, 0.56, 0.32)
);

float fogOf(float t) { return clamp(1.35 - 0.16 * t, 0.0, 1.0); }

// next intersection of the geodesic ray h(t) = cos(t)*uEye + sin(t)*v, for t > t0+eps.
// All crossings are computed from the original ray (no reparametrization, so the
// great circle never bends between layers).
float nextHit(float t0, vec4 v, out bool isWall, out int idx) {
    float t = 1e30;
    idx = -1;
    isWall = false;
    float teps = t0 + 0.0008;
    // walls: great 2-spheres x_i = 0;  uEye_i cos t + v_i sin t = 0, crossings every PI
    for (int i = 0; i < 4; ++i) {
        if (abs(v[i]) < 1e-12 && abs(uEye[i]) < 1e-5) continue;
        float b0 = atan(-uEye[i], v[i]);
        float z1 = (b0 > 0.0008) ? b0 : b0 + PI;
        float k = floor((teps - z1) / PI) + 1.0;
        float ti = z1 + k * PI;
        if (ti < t) { t = ti; isWall = true; idx = i; }
    }
    // moons: geodesic balls h . c >= cos(r);  surface crossings at phi +- th0 + 2pi*k
    for (int j = 0; j < 5; ++j) {
        float a = dot(uEye, moons[j].c);
        float b = dot(v, moons[j].c);
        float R = sqrt(a * a + b * b);
        float cr = cos(moons[j].r);
        if (R < cr - 1e-5) continue;
        float th0 = acos(clamp(cr / R, -1.0, 1.0));
        float phi = atan(b, a);
        float tEx = phi + th0;
        float tEn = phi - th0;
        float kE = floor((teps - tEx) / TWOPI) + 1.0;
        float kI = floor((teps - tEn) / TWOPI) + 1.0;
        float tau = min(tEx + kE * TWOPI, tEn + kI * TWOPI);
        if (tau < t) { t = tau; isWall = false; idx = j; }
    }
    return t;
}

vec3 shadeWall(vec4 h, vec4 v, int i, float t) {
    vec4 n4 = vec4(0.0);
    n4[i] = 1.0;
    n4 = normalize(n4 - dot(n4, h) * h);
    if (dot(n4, v) > 0.0) n4 = -n4;
    vec4 L = normalize(uSun - dot(uSun, h) * h);
    float dif = max(dot(n4, L), 0.0);
    vec3 alb = wallCol[i];
    float e = 1.0;
    for (int j = 0; j < 4; ++j) if (j != i) e = min(e, abs(h[j]));
    alb *= mix(0.35, 1.0, smoothstep(0.0, 0.05, e));   // darken the room edges
    return alb * (0.42 + 0.58 * dif) * fogOf(t);
}

vec3 shadeMoon(vec4 h, vec4 v, int j, float t) {
    vec4 n4 = normalize(h - cos(moons[j].r) * moons[j].c);
    if (dot(n4, v) > 0.0) n4 = -n4;
    vec4 L = normalize(uSun - dot(uSun, h) * h);
    float dif = max(dot(n4, L), 0.0);
    float rim = pow(clamp(1.0 - abs(dot(n4, v)), 0.0, 1.0), 3.0);
    float lat = acos(clamp(dot(h, moons[j].c), -1.0, 1.0));
    float band = 0.85 + 0.15 * sin(9.0 * lat);
    vec3 c = moons[j].col * (band * (0.26 + 0.85 * dif) + rim * 0.35 + 0.06);
    return c * fogOf(t);
}

void main() {
    vec2 ndc = vUV * 2.0 - 1.0;
    vec4 v = normalize(uF + uR * (ndc.x * uAspect * uTanF) + uU * (ndc.y * uTanF));
    vec3 acc = vec3(0.0);
    float trans = 1.0;
    float t0 = 0.0;
    for (int k = 0; k < 10; ++k) {
        bool isWall; int idx;
        float t = nextHit(t0, v, isWall, idx);
        if (t > 1e29) break;
        vec4 h = cos(t) * uEye + sin(t) * v;
        vec3 col = isWall ? shadeWall(h, v, idx, t) : shadeMoon(h, v, idx, t);
        float a = isWall ? uWallAlpha : 1.0;
        acc += trans * a * col;
        trans *= (1.0 - a);
        if (trans < 0.03) break;
        t0 = t + 0.0008;
    }
    vec3 bg = vec3(0.012, 0.015, 0.030);
    outColor = vec4(acc + trans * bg, 1.0);
}
)GLSL";

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "Shader compile error: %s\n", log);
        exit(1);
    }
    return s;
}

static void saveBMP(const char* path, int w, int h, const unsigned char* rgba) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    int row = ((w * 3 + 3) / 4) * 4;
    unsigned char hdr[54];
    memset(hdr, 0, sizeof(hdr));
    hdr[0] = 'B'; hdr[1] = 'M';
    int fsize = 54 + row * h;
    memcpy(hdr + 2, &fsize, 4);
    hdr[10] = 54;
    hdr[14] = 40;
    memcpy(hdr + 18, &w, 4);
    memcpy(hdr + 22, &h, 4);
    hdr[26] = 1;
    hdr[28] = 24;
    fwrite(hdr, 1, 54, f);
    unsigned char* r = new unsigned char[row];
    for (int y = 0; y < h; ++y) {
        memset(r, 0, row);
        for (int x = 0; x < w; ++x) {
            const unsigned char* p = rgba + ((long long)y * w + x) * 4;
            r[x * 3 + 0] = p[2];
            r[x * 3 + 1] = p[1];
            r[x * 3 + 2] = p[0];
        }
        fwrite(r, 1, row, f);
    }
    delete[] r;
    fclose(f);
}

static bool dragging = false;
static double mx = 0.0, my = 0.0;

static void cursorCB(GLFWwindow* w, double x, double y) {
    if (!dragging) { mx = x; my = y; return; }
    (void)w;
    float dx = (float)(x - mx), dy = (float)(y - my);
    mx = x; my = y;
    rotYaw(dx * 0.0035f);
    float pb = pitchTotal;
    pitchTotal += dy * 0.0035f;
    if (pitchTotal > 1.45f) pitchTotal = 1.45f;
    if (pitchTotal < -1.45f) pitchTotal = -1.45f;
    rotPitch(pitchTotal - pb);
}

static void mouseCB(GLFWwindow* w, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        dragging = (action == GLFW_PRESS);
        if (dragging) glfwGetCursorPos(w, &mx, &my);
    }
}

int main(int argc, char** argv) {
    bool shot = argc > 1 && strcmp(argv[1], "--shot") == 0;

    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (shot) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    const int W = 1280, H = 800;
    GLFWwindow* win = glfwCreateWindow(W, H, "3-Sphere Viewer", nullptr, nullptr);
    if (!win) { fprintf(stderr, "window creation failed\n"); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "glad failed to load OpenGL\n");
        return 1;
    }
    printf("OpenGL %s\n", (const char*)glGetString(GL_VERSION));

    GLuint vs = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        fprintf(stderr, "Program link error: %s\n", log);
        return 1;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(prog);

    GLint locEye = glGetUniformLocation(prog, "uEye");
    GLint locF   = glGetUniformLocation(prog, "uF");
    GLint locR   = glGetUniformLocation(prog, "uR");
    GLint locU   = glGetUniformLocation(prog, "uU");
    GLint locSun = glGetUniformLocation(prog, "uSun");
    GLint locAsp = glGetUniformLocation(prog, "uAspect");
    GLint locTan = glGetUniformLocation(prog, "uTanF");
    GLint locWal = glGetUniformLocation(prog, "uWallAlpha");

    V4 sun = vnorm(V4{ 0.35f, 0.85f, 1.1f, -0.25f });
    float wallAlpha = 0.25f;

    resetPose();
    glfwSetCursorPosCallback(win, cursorCB);
    glfwSetMouseButtonCallback(win, mouseCB);

    auto renderFrame = [&](int fw, int fh) {
        glViewport(0, 0, fw, fh);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUniform4f(locEye, eye.x, eye.y, eye.z, eye.w);
        glUniform4f(locF, fwd.x, fwd.y, fwd.z, fwd.w);
        glUniform4f(locR, rgt.x, rgt.y, rgt.z, rgt.w);
        glUniform4f(locU, up.x, up.y, up.z, up.w);
        glUniform4f(locSun, sun.x, sun.y, sun.z, sun.w);
        glUniform1f(locAsp, (float)fw / (float)fh);
        glUniform1f(locTan, 1.0913f);
        glUniform1f(locWal, wallAlpha);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    };

    if (shot) {
        bool bench = argc > 2 && strcmp(argv[2], "--bench") == 0;
        bool noWalls = argc > 2 && strcmp(argv[2], "--nowalls") == 0;
        bool testPose = argc > 2 && strcmp(argv[2], "--test") == 0;
        if (bench) {
            glfwSwapInterval(0);
            renderFrame(W, H);
            glfwSwapBuffers(win);
            glFinish();
            double t0 = glfwGetTime();
            for (int i = 0; i < 60; ++i) { renderFrame(W, H); glfwSwapBuffers(win); }
            glFinish();
            double t1 = glfwGetTime();
            printf("avg %.2f ms/frame (%.0f fps)\n", (t1 - t0) * 1000.0 / 60.0, 60.0 / (t1 - t0));
            return 0;
        }
        if (noWalls) wallAlpha = 0.0f;
        if (testPose && argc > 3) {
            int t = atoi(argv[3]);
            switch (t) {
            case 1: rotYaw(1.2f); break;
            case 2: rotPitch(0.8f); break;
            case 3: rotRoll(0.5f); rotPitch(0.6f); break;
            case 4: moveAlong(fwd, 1.0f); rotYaw(2.4f); break;
            default: break;
            }
        }
        renderFrame(W, H);
        unsigned char* px = new unsigned char[(long long)W * H * 4];
        glReadBuffer(GL_BACK);
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
        char name[64];
        snprintf(name, sizeof(name), "shot%s%s.bmp",
                 noWalls ? "_nw" : "", testPose ? "_t" : "");
        saveBMP(name, W, H, px);
        delete[] px;
        printf("saved %s\n", name);
        return 0;
    }

    printf("Controls: W/S forward/back | A/D strafe | R/F up/down | Q/E roll |\n");
    printf("          mouse-drag to look | SHIFT faster | M wall opacity | T reset | ESC quit\n");

    double lastT = glfwGetTime();
    bool prevM = false;
    int fc = 0;
    while (!glfwWindowShouldClose(win)) {
        double now = glfwGetTime();
        float dt = (float)(now - lastT);
        lastT = now;
        if (dt > 0.1f) dt = 0.1f;

        float sp = dt;
        if (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) sp *= 2.5f;

        if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) moveAlong(fwd, sp);
        if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) moveAlong(fwd, -sp);
        if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) moveAlong(rgt, -sp);
        if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) moveAlong(rgt, sp);
        if (glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS) moveAlong(up, sp);
        if (glfwGetKey(win, GLFW_KEY_F) == GLFW_PRESS) moveAlong(up, -sp);
        if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) rotRoll(+2.2f * dt);
        if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) rotRoll(-2.2f * dt);
        if (glfwGetKey(win, GLFW_KEY_M) == GLFW_PRESS) {
            if (!prevM) wallAlpha = (wallAlpha > 0.5f) ? 0.25f : (wallAlpha > 0.1f ? 0.0f : 0.92f);
            prevM = true;
        } else prevM = false;
        if (glfwGetKey(win, GLFW_KEY_T) == GLFW_PRESS) resetPose();
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(win, 1);

        int fw, fh;
        glfwGetFramebufferSize(win, &fw, &fh);
        renderFrame(fw, fh);
        glfwSwapBuffers(win);
        glfwPollEvents();

        if ((++fc % 15) == 0) {
            char title[256];
            snprintf(title, sizeof(title),
                     "3-Sphere Viewer | eye=(%+.2f %+.2f %+.2f %+.2f) room=(%c%c%c%c)",
                     eye.x, eye.y, eye.z, eye.w,
                     eye.x >= 0 ? '+' : '-', eye.y >= 0 ? '+' : '-',
                     eye.z >= 0 ? '+' : '-', eye.w >= 0 ? '+' : '-');
            glfwSetWindowTitle(win, title);
        }
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
