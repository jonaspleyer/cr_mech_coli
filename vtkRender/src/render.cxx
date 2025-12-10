#include <cmath>

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCompositeDataGeometryFilter.h>
#include <vtkCylinderSource.h>
#include <vtkExtractEdges.h>
#include <vtkGraphicsFactory.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkNamedColors.h>
#include <vtkOpenGLRenderer.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkPNGWriter.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSphereSource.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkWindowToImageFilter.h>
#include <vtkImageExport.h>

#include "render.h"

#define M_PI 3.14159265358979323846

namespace {
    vtkSmartPointer<vtkDataSet> CreateSphereDataSet(Vertex center, double radius)
    {
        double c[3] = {center.x, center.y, center.z};
        vtkNew<vtkSphereSource> sphere;
        sphere->SetCenter(c);
        sphere->SetRadius(radius);
        sphere->SetPhiResolution(30);
        sphere->SetThetaResolution(30);
        sphere->Update();
        return sphere->GetOutput();
    }

    vtkSmartPointer<vtkDataSet> CreateCylinderDataSet(
        double center[3],
        double direction[3],
        double radius,
        double height
    ) {
        double c[3] = {center[0], center[1], center[2]};
        vtkNew<vtkCylinderSource> cylinder;
        cylinder->SetHeight(height);
        cylinder->SetRadius(radius);
        cylinder->SetResolution(100);
        cylinder->Update();

        // Transform it
        vtkNew<vtkTransform> transform;
        transform->Identity();
        // Default orientation
        double orientation[3] = {0,1,0};
        double angle = std::acos(
            direction[0]*orientation[0] + direction[1]*orientation[1] + direction[2]*orientation[2]
        ) / 2 / M_PI * 360;
        double cross[3] = {
            orientation[1]*direction[2] - orientation[2]*direction[1],
            orientation[2]*direction[0] - orientation[0]*direction[2],
            orientation[0]*direction[1] - orientation[1]*direction[0]
        };
        transform->RotateWXYZ(angle, cross);
        transform->PostMultiply();
        transform->Translate(c);
        transform->Update();

        vtkNew<vtkTransformPolyDataFilter> transformPD;
        transformPD->SetTransform(transform);
        transformPD->SetInputData(cylinder->GetOutput());
        transformPD->SetInputConnection(cylinder->GetOutputPort());
        transformPD->Update();

        return transformPD->GetOutput();
    }
} // namespace

vtkNew<vtkMultiBlockDataSet> create_mesh(Agent agent) {
    // Create cell surfaces
    vtkNew<vtkMultiBlockDataSet> mesh;

    Vertex* vertices = agent.positions;
    double radius = agent.radius;

    mesh->SetBlock(0, CreateSphereDataSet(vertices[0], radius));
    for (int i=1; i<agent.n_vertices; i++) {
        double center[3] = {
            0.5 * (vertices[i].x + vertices[i-1].x),
            0.5 * (vertices[i].y + vertices[i-1].y),
            0.5 * (vertices[i].z + vertices[i-1].z)
        };
        double direction[3] = {
            vertices[i].x - vertices[i-1].x,
            vertices[i].y - vertices[i-1].y,
            vertices[i].z - vertices[i-1].z
        };
        double height = std::sqrt(
            (direction[0]*direction[0] + direction[1]*direction[1] + direction[2]*direction[2])
        );
        direction[0] /= height;
        direction[1] /= height;
        direction[2] /= height;
        mesh->SetBlock(2*i-1, CreateCylinderDataSet(center, direction, radius, height));
        mesh->SetBlock(2*i, CreateSphereDataSet(vertices[i], radius));
    }
    return mesh;
}

void render_img(Agent* agents, int n_agents, Camera camera, void* buffer)
{
    vtkNew<vtkGraphicsFactory> graphics_factory;
    graphics_factory->SetOffScreenOnlyMode(1);
    graphics_factory->SetUseMesaClasses(1);

    vtkNew<vtkRenderer> aren;
    aren->RemoveAllLights();
    aren->ResetCamera(0, camera.size_x, 0, camera.size_y, 0, camera.distance_z);
    aren->UseFXAAOff();
    aren->UseSSAOOff();

    auto cam = aren->GetActiveCamera();
    cam->SetParallelProjection(true);
    cam->SetPosition(camera.size_x/2.0, camera.size_y/2.0, camera.distance_z);
    cam->SetFocalPoint(camera.size_x/2.0, camera.size_y/2.0, 0);
    cam->OrthogonalizeViewUp();
    // cam->SetClippingRange(0, camera.distance_z);
    // cam->SetScreenBottomLeft(0,0,0);
    // cam->SetScreenTopRight(camera.size_x, camera.size_y, camera.distance_z);

    // double parallel_scale = max(hoiz_dist / aspect[0], vert_dist);
    // cam->SetParallelScale(parallel_scale);
    double scale = cam->GetParallelScale();
    cam->SetParallelScale(std::min(camera.size_x, camera.size_y) / 2);
    aren->ResetCameraClippingRange(0, camera.size_x, 0, camera.size_y, 0, camera.distance_z);

    // vtkNew<vtkRenderWindow> renWin;
    // renWin->AddRenderer(aren);

    for (int i=0; i<n_agents; i++) {
        auto mesh = create_mesh(agents[i]);

        vtkNew<vtkExtractEdges> edges;
        edges->SetInputData(mesh);

        vtkNew<vtkCompositeDataGeometryFilter> polydata;
        polydata->SetInputConnection(edges->GetOutputPort());

        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputConnection(0, polydata->GetOutputPort(0));

        vtkNew<vtkActor> actor;
        actor->SetMapper(mapper);
        actor->GetProperty()->SetLineWidth(2);
        actor->GetProperty()->SetColor(agents[i].color);// colors->GetColor3d("White").GetData());
        actor->UseBoundsOff();
        aren->AddActor(actor);
    }

    // aren->SetBackground();
    int size_x = camera.size_x * camera.resolution;
    int size_y = camera.size_y * camera.resolution;

    vtkNew<vtkRenderWindow> win;
    win->AddRenderer(aren);
    win->SetOffScreenRendering(1);
    win->SetSize(size_x, size_y);
    win->SetMultiSamples(0);
    win->Render();

    vtkNew<vtkWindowToImageFilter> windowToImageFilter;
    windowToImageFilter->SetInputBufferTypeToRGB();
    windowToImageFilter->SetInput(win);
    windowToImageFilter->Update();

    vtkNew<vtkImageExport> exporter;
    exporter->SetInputConnection(windowToImageFilter->GetOutputPort());
    // exporter->SetInputData(windowToImageFilter->GetOutput());
    exporter->Update();
    exporter->Export(buffer);

    int* dims = exporter->GetDataDimensions();

    std::vector<ssize_t> shape = { dims[2], dims[1], dims[0] };

    std::cout << dims[0] << ", " << dims[1] << ", " << dims[2] << "\n";
    // unsigned char* data = reinterpret_cast<unsigned char*>(malloc(dims[0]*dims[1]*dims[2]*sizeof(int)));
    std::cout << exporter->GetDataMemorySize() << "\n";

    /* std::cout << exporter->GetDataScalarTypeAsString() << "\n";

    for (int i=0; i<dims[0]; i++) {
        for (int j=0; j<dims[1]; j++) {
            for (int k=0; k<dims[2]; k++) {
                int index = i*dims[1]*dims[2] + j*dims[2] + k;
                unsigned char d[3] = {data[index]};

                if (d[0] != 0 || d[1] != 0 || d[2] != 0) {
                std::cout << i << ", " << j << ", " << k << ", " << index
                    << ", " << int(d[0]) << ", " << int(d[1]) << ", " << int(d[2]) << "\n";
                }
            }
        }
    }*/

    // vtkNew<vtkPNGWriter> writer;
    // writer->SetFileName("screenshot.png");
    // writer->SetInputConnection(windowToImageFilter->GetOutputPort());
    // writer->Write();
}

void example_usage() {
    double radius = 0.5;
    Vertex positions[4] = {
        Vertex {0.,0.,0.},
        Vertex {1.,1.,0.},
        Vertex {2.,2.1,0.},
        Vertex {2.8,3.0,0.}
    };

    Agent agent = Agent {
        positions: positions,
        radius: radius,
        n_vertices: 4,
        color: {160, 0, 0},
    };

    Vertex positions2[4] = {
        Vertex {3.0, 10.0},
        Vertex {4.0, 10.0},
        Vertex {5.0, 10.0},
        Vertex {15.0, 10.0}
    };
    Agent agent2 = Agent {
        positions: positions2,
        radius: radius,
        n_vertices: 4,
        color: {0, 150, 0},
    };

    Camera camera = Camera {
        size_x: 15.0,
        size_y: 10.0,
        distance_z: 2.0,
        resolution: 35.0,
    };
    Agent agents[] = {agent, agent2};
    // render_img(vertices, radius);
    // render_img(agents, 2, camera);
}
