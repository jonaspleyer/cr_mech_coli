use std::f32::consts::SQRT_2;

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
    pub interaction: RodInteraction<PhysicalInteraction>,
    /// Rate with which the cell grows in units `1/MIN`.
    #[pyo3(set, get)]
    pub growth_rate: f32,
    /// Threshold at which the cell will divide in units `MICROMETRE`.
    #[pyo3(set, get)]
    pub spring_length_threshold: f32,
    /// Reduces the growth rate with multiplier $((max - N)/max)^q $
    pub neighbor_reduction: Option<(usize, f32)>,
}

/// Describes all possible interaction variants
#[derive(Clone, Debug, Deserialize, Serialize)]
#[pyclass]
pub struct PhysicalInteraction(pub(crate) PhysInt, pub(crate) usize);

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) enum PhysInt {
    /// Wraps the :class:`MiePotentialF32`
    MiePotentialF32(MiePotentialF32),
    /// Wraps the :class:`MorsePotentialF32`
    MorsePotentialF32(MorsePotentialF32),
}

#[pymethods]
impl PhysicalInteraction {
    /// Constructs a new :class:`PhysicalInteraction`
    #[new]
    pub fn new(pyobject: Bound<PyAny>) -> PyResult<Self> {
        let mie_pot: Result<MiePotentialF32, _> = pyobject.extract();
        if let Ok(mie_pot) = mie_pot {
            return Ok(Self(PhysInt::MiePotentialF32(mie_pot), 0));
        };
        let morse_pot: Result<MorsePotentialF32, _> = pyobject.extract();
        if let Ok(morse_pot) = morse_pot {
            return Ok(Self(PhysInt::MorsePotentialF32(morse_pot), 0));
        }
        let pi: Result<PhysicalInteraction, _> = pyobject.extract();
        if let Ok(pi) = pi {
            return Ok(pi);
        }
        let ty_name = pyobject.get_type();
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Could not convert {ty_name} to any interaction potential. \
                Use one of the provided potentials instead.",
        )))
    }

    /// Extracts a copy of the inner value
    pub fn inner<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let res = match &self.0 {
            PhysInt::MiePotentialF32(mie) => Bound::new(py, mie.clone())?.into_any(),
            PhysInt::MorsePotentialF32(morse) => Bound::new(py, morse.clone())?.into_any(),
        };
        Ok(res)
    }

    /// Obtains the radius of the interaction
    #[getter]
    pub fn radius(&self) -> f32 {
        match &self.0 {
            PhysInt::MorsePotentialF32(m) => m.radius,
            PhysInt::MiePotentialF32(m) => m.radius,
        }
    }

    /// Getter for the strength of the interaction
    #[getter]
    pub fn strength(&self) -> f32 {
        match &self.0 {
            PhysInt::MorsePotentialF32(m) => m.strength,
            PhysInt::MiePotentialF32(m) => m.strength,
        }
    }

    /// Setter for the strength of the interaction
    #[setter]
    pub fn set_strength(&mut self, strength: f32) {
        match &mut self.0 {
            PhysInt::MorsePotentialF32(m) => m.strength = strength,
            PhysInt::MiePotentialF32(m) => m.strength = strength,
        }
    }

    /// Formats the [PhysicalInteraction]
    pub fn __repr__(&self) -> String {
        match &self.0 {
            PhysInt::MiePotentialF32(m) => format!("{:#?}", m),
            PhysInt::MorsePotentialF32(m) => format!("{:#?}", m),
        }
    }
}

type T = nalgebra::Vector3<f32>;

impl Interaction<T, T, T, f32> for PhysicalInteraction
where
    MorsePotentialF32: Interaction<T, T, T, f32>,
    MiePotentialF32: Interaction<T, T, T, f32>,
{
    fn calculate_force_between(
        &self,
        own_pos: &T,
        own_vel: &T,
        ext_pos: &T,
        ext_vel: &T,
        ext_info: &f32,
    ) -> Result<(T, T), CalcError> {
        use PhysInt::*;
        match &self.0 {
            MiePotentialF32(pot) => {
                pot.calculate_force_between(own_pos, own_vel, ext_pos, ext_vel, ext_info)
            }
            MorsePotentialF32(pot) => {
                pot.calculate_force_between(own_pos, own_vel, ext_pos, ext_vel, ext_info)
            }
        }
    }

    fn get_interaction_information(&self) -> f32 {
        match &self.0 {
            PhysInt::MiePotentialF32(pot) => <MiePotentialF32 as Interaction<
                nalgebra::Vector2<f32>,
                _,
                _,
                f32,
            >>::get_interaction_information(pot),
            PhysInt::MorsePotentialF32(pot) => <MorsePotentialF32 as Interaction<
                nalgebra::Vector2<f32>,
                _,
                _,
                f32,
            >>::get_interaction_information(pot),
        }
    }

    fn is_neighbor(&self, own_pos: &T, ext_pos: &T, ext_radius: &f32) -> Result<bool, CalcError> {
        for p in own_pos.row_iter() {
            for q in ext_pos.row_iter() {
                if (p - q).norm() < (self.radius() + ext_radius) / SQRT_2 {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn react_to_neighbors(&mut self, neighbors: usize) -> Result<(), CalcError> {
        self.1 = neighbors;
        Ok(())
    }
}

#[pymethods]
impl RodAgent {
    /// Constructs a new :class:`RodAgent`
    #[new]
    #[pyo3(signature = (
        pos,
        vel ,
        interaction,
        diffusion_constant=0.0,
        spring_tension=1.0,
        rigidity=2.0,
        spring_length=3.0,
        damping=1.0,
        growth_rate=0.1,
        spring_length_threshold=6.0,
        neighbor_reduction=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new<'py>(
        _py: Python<'py>,
        pos: numpy::PyReadonlyArray2<'py, f32>,
        vel: numpy::PyReadonlyArray2<'py, f32>,
        interaction: Bound<PyAny>,
        diffusion_constant: f32,
        spring_tension: f32,
        rigidity: f32,
        spring_length: f32,
        damping: f32,
        growth_rate: f32,
        spring_length_threshold: f32,
        neighbor_reduction: Option<(usize, f32)>,
    ) -> pyo3::PyResult<Self> {
        let pos = pos.as_array();
        let vel = vel.as_array();
        let nrows = pos.shape()[0];
        let pos = nalgebra::Matrix3xX::from_iterator(nrows, pos.to_owned());
        let vel = nalgebra::Matrix3xX::from_iterator(nrows, vel.to_owned());
        let interaction = PhysicalInteraction::new(interaction)?;
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
            interaction: RodInteraction(interaction),
            growth_rate,
            spring_length_threshold,
            neighbor_reduction,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self)
    }

    fn __deepcopy__(&self, _memo: pyo3::Bound<pyo3::types::PyDict>) -> Self {
        self.clone()
    }

    /// Position of the agent given by a matrix containing all vertices in order.
    #[getter]
    pub fn pos<'a>(&'a self, py: Python<'a>) -> Bound<'a, numpy::PyArray2<f32>> {
        use numpy::ToPyArray;
        self.mechanics.pos.clone().to_pyarray(py)
    }

    /// Position of the agent given by a matrix containing all vertices in order.
    #[setter]
    pub fn set_pos<'a>(&'a mut self, pos: Bound<'a, numpy::PyArray2<f32>>) -> pyo3::PyResult<()> {
        use numpy::PyArrayMethods;
        // TODO check this function: I think this produces an error.
        let iter: Vec<f32> = pos.to_vec()?;
        self.mechanics.pos =
            nalgebra::MatrixXx3::<f32>::from_iterator(self.mechanics.pos.nrows(), iter);
        Ok(())
    }

    /// Velocity of the agent given by a matrix containing all velocities at vertices in order.
    #[getter]
    pub fn vel<'a>(&'a self, py: Python<'a>) -> Bound<'a, numpy::PyArray2<f32>> {
        use numpy::ToPyArray;
        numpy::nalgebra::MatrixXx3::from(self.mechanics.vel.clone()).to_pyarray(py)
    }

    /// Velocity of the agent given by a matrix containing all velocities at vertices in order.
    #[setter]
    pub fn set_vel<'a>(&'a mut self, pos: Bound<'a, numpy::PyArray2<f32>>) -> pyo3::PyResult<()> {
        use numpy::PyArrayMethods;
        let iter: Vec<f32> = pos.to_vec()?;
        self.mechanics.vel = nalgebra::MatrixXx3::<f32>::from_iterator(iter.len(), iter);
        Ok(())
    }

    /// The interaction radius as given by the [MorsePotentialF32] interaction struct.
    #[getter]
    pub fn radius(&self) -> f32 {
        match &self.interaction.0 .0 {
            PhysInt::MorsePotentialF32(pot) => pot.radius,
            PhysInt::MiePotentialF32(pot) => pot.radius,
        }
    }
}

impl Cycle<RodAgent, f32> for RodAgent {
    fn update_cycle(
        _rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f32,
        cell: &mut Self,
    ) -> Option<CycleEvent> {
        let rate = if let Some((max, exp)) = cell.neighbor_reduction {
            let m = max as f32;
            let n = cell.interaction.0 .1 as f32;
            cell.growth_rate * ((m - n) / m).max(0.).powf(exp)
        } else {
            cell.growth_rate
        };
        cell.mechanics.spring_length += rate * dt;
        if cell.mechanics.spring_length > cell.spring_length_threshold {
            Some(CycleEvent::Division)
        } else {
            None
        }
    }

    fn divide(_rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Self) -> Result<Self, DivisionError> {
        let c2_mechanics = cell.mechanics.divide(cell.radius())?;
        let mut c2 = cell.clone();
        c2.mechanics = c2_mechanics;
        Ok(c2)
    }
}
