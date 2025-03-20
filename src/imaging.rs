use pyo3::prelude::*;

/// Converts an integer counter between 0 and 251^3-1 to an RGB value.
/// The reason why 251 was chosen is due to the fact that it is the highest prime number which is
/// below 255.
/// This will yield a Field of numbers :math:`\mathbb{Z}/251 \mathbb{Z}` and thus we will be able
/// to determine an exact inverse function.
/// This system is bettern known as
/// `modular arithmetic <https://en.wikipedia.org/wiki/Modular_arithmetic>`_.
///
/// To calculate artistic color values we multiply the counter by 157*163*173 which are three prime
/// numbers roughyl in the middle of 255.
/// The new numbers can be calculated via
///
/// >>> new_counter = counter * 157 * 163 * 173
/// >>> c1, mod = divmod(new_counter, 251**2)
/// >>> c2, mod = divmod(mod, 251)
/// >>> c3      = mod
///
/// Args:
///     counter (int): Counter between 0 and 251^3-1
///     artistic (bool): Enables artistic to provide larger differences between single steps instead
///         of simple incremental one.
///
/// Returns:
///     list[int]: A list with exactly 3 entries containing the calculated color.
#[pyfunction]
pub fn counter_to_color(counter: u32) -> (u8, u8, u8) {
    let mut counter: u128 = counter as u128;
    counter = (counter * 157 * 163 * 173) % 251u128.pow(3);
    let mut color = (0, 0, 0);
    let (q, m) = num::Integer::div_mod_floor(&counter, &251u128.pow(2));
    color.0 = q as u8;
    let (q, m) = num::Integer::div_mod_floor(&m, &251);
    color.1 = q as u8;
    color.2 = m as u8;
    color
}

/// Converts a given color back to the counter value.
///
/// The is the inverse of the :func:`counter_to_color` function.
/// The formulae can be calculated with the `Extended Euclidean Algorithm
/// <https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm>`_.
/// The multiplicative inverse (mod 251) of the numbers of 157, 163 and 173 are:
///
/// >>> assert (12590168 * 157) % 251**3 == 1
/// >>> assert (13775961 * 163) % 251**3 == 1
/// >>> assert (12157008 * 173) % 251**3 == 1
///
/// Thus the formula to calculate the counter from a given color is:
///
/// >>> counter = color[0] * 251**2 + color[1] * 251 + color[2]
/// >>> counter = (counter * 12590168) % 251**3
/// >>> counter = (counter * 13775961) % 251**3
/// >>> counter = (counter * 12157008) % 251**3
#[pyfunction]
pub fn color_to_counter(color: (u8, u8, u8)) -> u32 {
    let mut counter: u128 =
        color.0 as u128 * 251u128.pow(2) + color.1 as u128 * 251u128.pow(1) + color.2 as u128;
    counter = (counter * 12590168) % 251u128.pow(3);
    counter = (counter * 13775961) % 251u128.pow(3);
    counter = (counter * 12157008) % 251u128.pow(3);
    counter as u32
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_color_counter_conversion() {
        for i in 1..251u32.pow(3) {
            let color = counter_to_color(i);
            let counter = color_to_counter(color);
            assert_eq!(i, counter);
        }
    }

    #[test]
    fn test_counter_to_color_conversion() {
        for i in 1..251u8 {
            for j in 1..251u8 {
                for k in 1..251u8 {
                    let color = (i, j, k);
                    let counter = color_to_counter(color);
                    let color_back = counter_to_color(counter);
                    assert_eq!(color, color_back);
                }
            }
        }
    }
}
