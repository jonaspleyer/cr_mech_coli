import cr_mech_coli as crm


def test_create_potentials():
    p1 = crm.MorsePotentialF32(
        radius=1.0,
        potential_stiffness=0.5,
        cutoff=3.0,
        strength=0.1,
    )
    p2 = crm.MiePotentialF32(
        radius=1.0,
        strength=0.1,
        bound=3.0,
        cutoff=3.0,
        en=2.0,
        em=1.0,
    )
    pot1 = crm.PhysicalInteraction(p1)
    assert pot1.radius == 1.0
    assert pot1.cutoff == 3.0
    assert pot1.potential_stiffness == 0.5

    pot2 = crm.PhysicalInteraction(p2)
    assert pot2.radius == 1.0
    assert pot2.cutoff == 3.0
    assert pot2.en == 2.0
    assert pot2.em == 1.0


if __name__ == "__main__":
    test_create_potentials()
