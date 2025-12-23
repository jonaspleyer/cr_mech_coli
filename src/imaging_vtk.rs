use cellular_raza::prelude::CellIdentifier;
use pyo3::prelude::*;
use std::collections::HashMap;

/// asdf
#[repr(C)]
pub struct Vertex {
    /// x coordinate
    pub x: f64,
    /// y coordinate
    pub y: f64,
    /// z coordinate
    pub z: f64,
}

/// asdf
#[repr(C)]
pub struct Agent {
    /// vertices of the agent
    positions: *const Vertex,
    /// Thickness of the rod
    radius: f64,
    /// number of vertices
    n_vertices: std::ffi::c_int,
    /// color of the rod
    color: [f64; 3],
}

impl Agent {
    /// Creates a new [Agent] used for plotting
    pub fn new(positions: &[Vertex], radius: f64, color: [f64; 3]) -> Agent {
        let n_vertices = positions.len() as i32;
        let positions = positions.as_ptr();
        Agent {
            positions,
            radius,
            color,
            n_vertices,
        }
    }
}

/// asdf
#[repr(C)]
pub struct Camera {
    /// x-dimension of the scene
    pub size_x: f64,
    /// y-dimension of the scene
    pub size_y: f64,
    /// distance of camera to middle of scene
    pub distance_z: f64,
    /// resolution of image
    pub resolution: f64,
}

/// Renders a set of agents with camera settings
pub fn render_agents(
    agents: &[Agent],
    camera: Camera,
) -> Result<ndarray::Array3<u8>, ndarray::ShapeError> {
    let n_agents = agents.len();
    let agents = agents.as_ptr();

    let size_x = (camera.size_x * camera.resolution) as usize;
    let size_y = (camera.size_y * camera.resolution) as usize;

    // let mut buffer: Vec<u8> = vec![0; size_x * size_y * 3];
    let mut buffer = vec![0u8; size_x * size_y * 3];

    unsafe {
        let buf_ptr = buffer.as_mut_ptr() as *mut std::ffi::c_void;
        render_img(agents, n_agents, camera, buf_ptr);
    }

    ndarray::Array3::from_shape_vec((size_x, size_y, 3), buffer)
}

/// asdf
#[pyfunction]
pub fn render_mask_rs<'py>(
    py: Python<'py>,
    cells: HashMap<CellIdentifier, (crate::RodAgent, Option<CellIdentifier>)>,
    cell_to_color: HashMap<CellIdentifier, (u8, u8, u8)>,
    domain_size: (f32, f32),
    resolution: f32,
) -> pyo3::PyResult<Bound<'py, numpy::PyArray3<u8>>> {
    use numpy::*;

    let camera = Camera {
        size_x: domain_size.0 as f64,
        size_y: domain_size.1 as f64,
        distance_z: domain_size.0.max(domain_size.1) as f64,
        resolution: resolution as f64,
    };

    let agents = cells
        .iter()
        .map(|(ident, (c, _))| {
            let color = cell_to_color[ident];
            let color = [color.0 as f64, color.1 as f64, color.2 as f64];
            let positions = c
                .mechanics
                .pos
                .row_iter()
                .map(|row| Vertex {
                    x: row[0] as f64,
                    y: row[1] as f64,
                    z: row[2] as f64,
                })
                .collect::<Vec<_>>();
            let radius = c.interaction.0.radius() as f64;
            Agent::new(&positions, radius, color)
        })
        .collect::<Vec<_>>();

    let res = render_agents(&agents, camera)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

    Ok(res.to_pyarray(py))
}

unsafe extern "C" {
    fn render_img(
        agents: *const Agent,
        n_agents: usize,
        camera: Camera,
        buffer: *mut std::ffi::c_void,
    );
}
