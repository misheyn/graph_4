#include <iostream>
#include <cstdlib>
#include <GL/freeglut.h>

#define PART_COUNT 1000
clock_t startTime;

typedef struct particle {
    double posX;
    double posX0;
    double posY;
    double posY0;
    double vx;
    double q;
    double m;
} particle;

particle *part = nullptr;
particle *partGPU = nullptr;
int *sizeGPU = nullptr;
clock_t *timeGPU = nullptr;

void moveParticle(particle *arr, int size, clock_t time) {
    for (int i = 0; i < size; ++i) {
        if (arr[i].posX > -0.5)
            arr[i].posY = arr[i].posY0 +
                          (arr[i].q * 0.001 * ((double) time / 10000.0) * ((double) time / 10000.0)) / (2 * arr[i].m);
        arr[i].posX = arr[i].posX0 + arr[i].vx * (double) time / 10000.0;
    }
}

__global__ void moveParticleGPU(particle *arr, int *size, clock_t *time) {
    unsigned th = threadIdx.x;
    unsigned bl = blockIdx.x;
    unsigned i = bl * 1024 + th;
    if (i < *size) {
        if (arr[i].posX > -0.5)
            arr[i].posY = arr[i].posY0 +
                          (arr[i].q * 0.001 * ((double) *time / 10000.0) * ((double) *time / 10000.0)) / (2 * arr[i].m);
        arr[i].posX = arr[i].posX0 + arr[i].vx * (double) *time / 10000.0;
    }
}

void initParticles() {
    part = (particle *) malloc(sizeof(particle) * PART_COUNT);
    cudaMalloc((void **)&partGPU, sizeof(particle) * PART_COUNT);
    cudaMalloc((void **)&sizeGPU, sizeof(int));
    cudaMalloc((void **)&timeGPU, sizeof(clock_t));

    for (int i = 0; i < PART_COUNT; i++) {
        part[i].posY0 = (-30 + rand() % 60) / 100.0;
        part[i].posX0 = -1;
        part[i].vx = (10 + rand() % 200) / 2000.0;
        part[i].posX = part[i].posX0;
        part[i].posY = part[i].posY0;
        part[i].m = (rand() % 1000) / 10000000000.0;
        part[i].q = (-1000 + rand() % 2000) / 10000000000.0;
    }
}

void display() {
    int size = PART_COUNT;
    clock_t time = clock() - startTime;
    unsigned threadX, blockX;

    glPushMatrix();
    glEnable(GL_POINT_SMOOTH);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (size > 1024) {
        threadX = 1024;
        blockX = 1 + (size) / 1024;
    } else {
        threadX = size;
        blockX = 1;
    }
    cudaMemcpy(partGPU, part, sizeof (particle) * PART_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(sizeGPU, &size, sizeof (int), cudaMemcpyHostToDevice);
    cudaMemcpy(timeGPU, &time, sizeof (clock_t), cudaMemcpyHostToDevice);

    moveParticleGPU<<<blockX, threadX>>>(partGPU, sizeGPU, timeGPU);

    cudaDeviceSynchronize();
    cudaMemcpy(part, partGPU, sizeof (particle) * PART_COUNT, cudaMemcpyDeviceToHost);

    glBegin(GL_LINES);
    glColor3d(1, 1, 1);
    glVertex2d(-0.5, -1);
    glVertex2d(-0.5, -0.3);
    glVertex2d(-0.5, 0.3);
    glVertex2d(-0.5, 1);
    glEnd();

    glBegin(GL_POINTS);
    for (int i = 0; i < PART_COUNT; ++i) {
        if (part[i].q < 0)
            glColor3d(0, 0, 1);
        else
            glColor3d(0, 1, 0);
        glVertex2d(part[i].posX, part[i].posY);
    }
    glEnd();

    glPopMatrix();
    glutSwapBuffers();
}

void timer() {
    glutPostRedisplay();
    glutTimerFunc(10, reinterpret_cast<void (*)(int)>(timer), 0);
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    initParticles();
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(900, 700);
    glutInitWindowPosition(450, 60);
    glutCreateWindow("Particles in an electric field");
    glClearColor(0, 0, 0, 0);
    glutDisplayFunc(display);
    glutTimerFunc(10, reinterpret_cast<void (*)(int)>(timer), 0);
    startTime = clock();
    glutMainLoop();
}
