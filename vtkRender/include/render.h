// #include "render.h"

#define M_PI 3.14159265358979323846

extern "C" struct Vertex {
    double x;
    double y;
    double z;
};

// Generating meshes for agents
extern "C" struct Agent {
    Vertex* positions;
    double radius;
    int n_vertices;
    double color[3];
};

// Camera placement
extern "C" struct Camera {
    double size_x;
    double size_y;
    double distance_z;
    double resolution;
};

extern "C" void example_usage();

extern "C" void render_img(Agent* agents, int n_agents, Camera camera, void* buffer);
