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
    qot1 = pot1.inner()
    assert qot1.radius == 1.0
    assert qot1.cutoff == 3.0
    assert qot1.potential_stiffness == 0.5
    pot2 = crm.PhysicalInteraction(p2)
    qot2 = pot2.inner()
    assert qot2.radius == 1.0
    assert qot2.cutoff == 3.0
    assert qot2.en == 2.0
    assert qot2.em == 1.0


if __name__ == "__main__":
    test_create_potentials()
