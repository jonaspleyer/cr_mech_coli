use std::f32::consts::SQRT_2;

use approx::AbsDiffEq;
use cellular_raza::prelude::*;
use pyo3::{prelude::*, types::PyDict, IntoPyObjectExt};
use serde::{Deserialize, Serialize};

/// Defines how the growth rates of the daughter cells will be set
#[pyclass]
#[derive(Clone, Debug, Deserialize, Serialize, AbsDiffEq, PartialEq)]
pub enum GrowthRateSetter {
    /// Pick the new growth rates from a normal distribution
    NormalDistr {
        /// Mean of the distribution
        mean: f32,
        /// Standard deviation of the distribution
        std: f32,
    },
    /// Explicitly define the two new growth rates
    Explicit {
        /// Growth rate 1
        g1: f32,
        /// Growth rate 2
        g2: f32,
    },
}

impl GrowthRateSetter {
    /// Creates an instance of the object from a given python dict.
    ///
    /// The dictionary must match exactly with no additional keys.
    pub(crate) fn from_pydict(dict: &Bound<PyDict>) -> PyResult<Self> {
        let extract_key = |key: &str| -> PyResult<Option<f32>> {
            if let Some(item) = dict.get_item(key)? {
                Ok(Some(item.extract()?))
            } else {
                Ok(None)
            }
        };
        let mean = extract_key("mean")?;
        let std = extract_key("std")?;
        let g1 = extract_key("g1")?;
        let g2 = extract_key("g2")?;
        match (mean, std, g1, g2) {
            (Some(mean), Some(std), None, None) => Ok(GrowthRateSetter::NormalDistr { mean, std }),
            (None, None, Some(g1), Some(g2)) => Ok(GrowthRateSetter::Explicit { g1, g2 }),
            _ => Err(pyo3::exceptions::PyKeyError::new_err("could not find suitable combination of either ('mean', 'std') or ('g1', 'g2') keys in dict.")),
        }
    }

    pub(crate) fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        use pyo3::types::IntoPyDict;
        use GrowthRateSetter::*;
        match self {
            NormalDistr { mean, std } => [
                (
                    "mean",
                    pyo3::types::PyFloat::new(py, *mean as f64).into_py_any(py)?,
                ),
                (
                    "std",
                    pyo3::types::PyFloat::new(py, *std as f64).into_py_any(py)?,
                ),
            ],
            Explicit { g1, g2 } => [
                (
                    "g1",
                    pyo3::types::PyFloat::new(py, *g1 as f64).into_py_any(py)?,
                ),
                (
                    "g2",
                    pyo3::types::PyFloat::new(py, *g2 as f64).into_py_any(py)?,
                ),
            ],
        }
        .into_py_dict(py)
    }

    unsafe fn default_dict() -> Bound<'static, PyDict> {
        let setter = Self::default();
        let py = Python::assume_attached();
        setter.to_pydict(py).unwrap()
    }
}

#[pymethods]
impl GrowthRateSetter {
    #[new]
    #[pyo3(signature=(kwds))]
    fn new(kwds: Bound<PyDict>) -> PyResult<Self> {
        Self::from_pydict(&kwds)
    }

    /// Get attributes of the class
    pub fn __getattr__(&self, name: &str) -> pyo3::PyResult<f32> {
        let e = Err(pyo3::exceptions::PyValueError::new_err(format!(
            "GrowthRateSetter does not have attribute '{name}'"
        )));
        match self {
            GrowthRateSetter::NormalDistr { mean, std } => match name {
                "mean" => Ok(*mean),
                "std" => Ok(*std),
                _ => e,
            },
            GrowthRateSetter::Explicit { g1, g2 } => match name {
                "g1" => Ok(*g1),
                "g2" => Ok(*g2),
                _ => e,
            },
        }
    }
}

impl Default for GrowthRateSetter {
    fn default() -> Self {
        Self::NormalDistr {
            mean: 0.01,
            std: 0.0,
        }
    }
}

/// A basic cell-agent which makes use of
/// `RodMechanics <https://cellular-raza.com/docs/cellular_raza_building_blocks/structs.RodMechanics.html>`_
#[pyclass]
#[derive(CellAgent, Clone, Debug, Deserialize, Serialize, AbsDiffEq, PartialEq)]
#[approx(epsilon_type = f32)]
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
    /// Determines the mean and width of distribution for sampling new values of growth rate.
    #[pyo3(set, get)]
    pub growth_rate_setter: GrowthRateSetter,
    /// Threshold at which the cell will divide in units `MICROMETRE`.
    #[pyo3(set, get)]
    pub spring_length_threshold: f32,
    /// Reduces the growth rate with multiplier $((max - N)/max)^q $
    #[pyo3(set, get)]
    #[approx(epsilon_map = |x| (x, x))]
    #[approx(map = |x: &Option<(usize, f32)>| x.map(|z| (z.0 as f32, z.1)))]
    pub neighbor_reduction: Option<(usize, f32)>,
}

/// Describes all possible interaction variants
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, AbsDiffEq)]
#[pyclass]
#[approx(epsilon_type = f32)]
#[serde(from = "PhysicalInteractionSerde")]
#[serde(into = "PhysicalInteractionSerde")]
pub struct PhysicalInteraction(pub PhysInt, #[approx(equal)] pub usize);

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PhysicalInteractionSerde {
    interaction: PhysInt,
    n_neighbors: usize,
}

impl From<PhysicalInteraction> for PhysicalInteractionSerde {
    fn from(value: PhysicalInteraction) -> Self {
        PhysicalInteractionSerde {
            interaction: value.0,
            n_neighbors: value.1,
        }
    }
}

impl From<PhysicalInteractionSerde> for PhysicalInteraction {
    fn from(value: PhysicalInteractionSerde) -> Self {
        PhysicalInteraction(value.interaction, value.n_neighbors)
    }
}

/// Contains all possible Intercation variants.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, AbsDiffEq)]
pub enum PhysInt {
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

impl InteractionInformation<f32> for PhysicalInteraction {
    fn get_interaction_information(&self) -> f32 {
        match &self.0 {
            PhysInt::MiePotentialF32(pot) => {
                <MiePotentialF32 as InteractionInformation<f32>>::get_interaction_information(pot)
            }
            PhysInt::MorsePotentialF32(pot) => {
                <MorsePotentialF32 as InteractionInformation<f32>>::get_interaction_information(pot)
            }
        }
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
        growth_rate=0.01,
        growth_rate_setter=GrowthRateSetter::default_dict(),
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
        growth_rate_setter: Bound<'py, PyDict>,
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
            growth_rate_setter: GrowthRateSetter::from_pydict(&growth_rate_setter)?,
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
        // Determine growth rate depening on number of neighbors
        let rate = if let Some((max, exp)) = cell.neighbor_reduction {
            let m = max as f32;
            let n = cell.interaction.0 .1 as f32;
            cell.growth_rate * ((m - n) / m).max(0.).powf(exp)
        } else {
            cell.growth_rate
        };

        // Exponential Growth
        cell.mechanics.spring_length += rate * dt * cell.mechanics.spring_length;
        if cell.mechanics.spring_length > cell.spring_length_threshold {
            Some(CycleEvent::Division)
        } else {
            None
        }
    }

    fn divide(rng: &mut rand_chacha::ChaCha8Rng, cell: &mut Self) -> Result<Self, DivisionError> {
        use rand_distr::Distribution;
        let c2_mechanics = cell.mechanics.divide(cell.radius())?;
        let mut c2 = cell.clone();
        // Pick new growth parameters
        let (g1, g2) = match cell.growth_rate_setter {
            GrowthRateSetter::NormalDistr { mean, std } => {
                let distr = rand_distr::Normal::new(mean, std)
                    .map_err(|e| DivisionError(format!("{e}")))?;
                (distr.sample(rng), distr.sample(rng))
            }
            GrowthRateSetter::Explicit { g1, g2 } => (g1, g2),
        };
        cell.growth_rate = g1;
        c2.growth_rate = g2;
        c2.mechanics = c2_mechanics;
        Ok(c2)
    }
}
