use cellular_raza::prelude::*;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// A basic cell-agent which makes use of
/// `RodMechanics <https://cellular-raza.com/docs/cellular_raza_building_blocks/structs.RodMechanics.html>`_
#[pyclass]
#[derive(CellAgent, Clone, Debug, Deserialize, Serialize)]
pub struct RodAgent {
    /// Determines mechanical properties of the agent.
    /// See :class:`RodMechanics`.
    #[Mechanics]
    pub mechanics: RodMechanics<f32, 3>,
    /// Determines interaction between agents. See [MorsePotentialF32].
    #[Interaction]
    pub interaction: RodInteraction<MorsePotentialF32>,
    /// Rate with which the cell grows in units `1/MIN`.
    #[pyo3(set, get)]
    pub growth_rate: f32,
    /// Threshold at which the cell will divide in units `MICROMETRE`.
    #[pyo3(set, get)]
    pub spring_length_threshold: f32,
}

#[pymethods]
impl RodAgent {
    /// Constructs a new :class:`RodAgent`
    #[new]
    #[pyo3(signature = (
        pos,
        vel ,
        diffusion_constant=0.0,
        spring_tension=1.0,
        rigidity=2.0,
        spring_length=3.0,
        damping=1.0,
        radius=3.0,
        strength=0.1,
        potential_stiffness=0.5,
        cutoff=10.0,
        growth_rate=0.1,
        spring_length_threshold=6.0,
    ))]
    pub fn new<'py>(
        _py: Python<'py>,
        pos: numpy::PyReadonlyArray2<'py, f32>,
        vel: numpy::PyReadonlyArray2<'py, f32>,
        diffusion_constant: f32,
        spring_tension: f32,
        rigidity: f32,
        spring_length: f32,
        damping: f32,
        radius: f32,
        strength: f32,
        potential_stiffness: f32,
        cutoff: f32,
        growth_rate: f32,
        spring_length_threshold: f32,
    ) -> pyo3::PyResult<Self> {
        let pos = pos.as_array();
        let vel = vel.as_array();
        let nrows = pos.shape()[0];
        let pos = nalgebra::Matrix3xX::from_iterator(nrows, pos.to_owned().into_iter());
        let vel = nalgebra::Matrix3xX::from_iterator(nrows, vel.to_owned().into_iter());
        Ok(Self {
            mechanics: RodMechanics {
                pos: pos.transpose(),
                vel: vel.transpose(),
                diffusion_constant,
                spring_tension,
                rigidity,
                spring_length,
                damping,
            },
            interaction: RodInteraction(MorsePotentialF32 {
                radius,
                strength,
                potential_stiffness,
                cutoff,
            }),
            growth_rate,
            spring_length_threshold,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __deepcopy__(&self, _memo: pyo3::Bound<pyo3::types::PyDict>) -> Self {
        self.clone()
    }

    /// Position of the agent given by a matrix containing all vertices in order.
    #[getter]
    pub fn pos<'a>(&'a self, py: Python<'a>) -> Bound<'a, numpy::PyArray2<f32>> {
        use numpy::ToPyArray;
        let new_array =
            numpy::nalgebra::Matrix::<f32, numpy::nalgebra::Dyn, numpy::nalgebra::U3, _>::from(
                self.mechanics.pos.clone(),
            );
        new_array.to_pyarray_bound(py)
    }

    /// Position of the agent given by a matrix containing all vertices in order.
    #[setter]
    pub fn set_pos<'a>(&'a mut self, pos: Bound<'a, numpy::PyArray2<f32>>) -> pyo3::PyResult<()> {
        use numpy::PyArrayMethods;
        let iter: Vec<f32> = pos.to_vec()?;
        self.mechanics.pos =
            nalgebra::MatrixXx3::<f32>::from_iterator(self.mechanics.pos.nrows(), iter.into_iter());
        Ok(())
    }

    /// Velocity of the agent given by a matrix containing all velocities at vertices in order.
    #[getter]
    pub fn vel<'a>(&'a self, py: Python<'a>) -> Bound<'a, numpy::PyArray2<f32>> {
        use numpy::ToPyArray;
        numpy::nalgebra::MatrixXx3::from(self.mechanics.vel.clone()).to_pyarray_bound(py)
    }

    /// Velocity of the agent given by a matrix containing all velocities at vertices in order.
    #[setter]
    pub fn set_vel<'a>(&'a mut self, pos: Bound<'a, numpy::PyArray2<f32>>) -> pyo3::PyResult<()> {
        use numpy::PyArrayMethods;
        let iter: Vec<f32> = pos.to_vec()?;
        self.mechanics.vel =
            nalgebra::MatrixXx3::<f32>::from_iterator(iter.len(), iter.into_iter());
        Ok(())
    }

    /// The interaction radius as given by the [MorsePotentialF32] interaction struct.
    #[getter]
    pub fn radius(&self) -> f32 {
        self.interaction.0.radius
    }
}

impl Cycle<RodAgent, f32> for RodAgent {
    fn update_cycle(
        _rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f32,
        cell: &mut Self,
    ) -> Option<CycleEvent> {
        cell.mechanics.spring_length += cell.growth_rate * dt;
        if cell.mechanics.spring_length > cell.spring_length_threshold {
            Some(CycleEvent::Division)
        } else {
            None
        }
    }

    fn divide(_rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Self) -> Result<Self, DivisionError> {
        let c2_mechanics = cell.mechanics.divide(cell.interaction.0.radius)?;
        let mut c2 = cell.clone();
        c2.mechanics = c2_mechanics;
        Ok(c2)
    }
}
