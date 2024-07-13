import jax.numpy as jnp

tiny = 1e-20


# @reaction('k01', [   (1,HI),   (1,de)], [  (1,HII),   (2,de)])
def rxn_k01(tev, tK, logtev):
    vals = jnp.exp(
        -32.71396786375
        + 13.53655609057 * logtev
        - 5.739328757388 * logtev**2
        + 1.563154982022 * logtev**3
        - 0.2877056004391 * logtev**4
        + 0.03482559773736999 * logtev**5
        - 0.00263197617559 * logtev**6
        + 0.0001119543953861 * logtev**7
        - 2.039149852002e-6 * logtev**8
    )
    return vals


# -- k02 --


# @reaction('k02', [  (1,HII),   (1,de)], [   (1,HI),         ])
def rxn_k02(tev, tK, logtev):
    _i1 = tK > 5500
    _i2 = ~_i1
    vals = jnp.exp(
        -28.61303380689232
        - 0.7241125657826851 * logtev
        - 0.02026044731984691 * logtev**2
        - 0.002380861877349834 * logtev**3
        - 0.0003212605213188796 * logtev**4
        - 0.00001421502914054107 * logtev**5
        + 4.989108920299513e-6 * logtev**6
        + 5.755614137575758e-7 * logtev**7
        - 1.856767039775261e-8 * logtev**8
        - 3.071135243196595e-9 * logtev**9
    )
    _i1 = tev < 0.8
    vals.at[_i2].set(
        1.54e-9
        * (1.0 + 0.3 / jnp.exp(8.099328789667 / tev[_i2]))
        / (jnp.exp(40.49664394833662 / tev[_i2]) * tev[_i2] ** 1.5)
        + 3.92e-13 / tev[_i2] ** 0.6353
    )
    vals.at[_i1].set(3.92e-13 / tev[_i1] ** 0.6353)
    return vals


# -- k03 --
# @reaction('k03', [  (1,HeI),   (1,de)], [ (1,HeII),   (2,de)])
def rxn_k03(tev, tK, logtev):
    _i1 = tev > 0.8
    _i2 = ~_i1
    vals = jnp.exp(
        -44.09864886561001
        + 23.91596563469 * logtev
        - 10.75323019821 * logtev**2
        + 3.058038757198 * logtev**3
        - 0.5685118909884001 * logtev**4
        + 0.06795391233790001 * logtev**5
        - 0.005009056101857001 * logtev**6
        + 0.0002067236157507 * logtev**7
        - 3.649161410833e-6 * logtev**8
    )
    vals.at[_i2].set(tiny)
    return vals


# -- k04 --


# @reaction('k04', [ (1,HeII),   (1,de)], [  (1,HeI),         ])
def rxn_k04(tev, tK, logtev):
    _i1 = tev > 0.8
    _i2 = ~_i1
    vals = (
        1.54e-9
        * (1.0 + 0.3 / jnp.exp(8.099328789667 / tev))
        / (jnp.exp(40.49664394833662 / tev) * tev**1.5)
        + 3.92e-13 / tev**0.6353
    )
    vals.at[_i2].set(tiny)
    return vals


# -- k05 --
# @reaction('k05', [ (1,HeII),   (1,de)], [(1,HeIII),   (2,de)])
def rxn_k05(tev, tK, logtev):
    _i1 = tev > 0.8
    _i2 = ~_i1
    vals = jnp.exp(
        -68.71040990212001
        + 43.93347632635 * logtev
        - 18.48066993568 * logtev**2
        + 4.701626486759002 * logtev**3
        - 0.7692466334492 * logtev**4
        + 0.08113042097303 * logtev**5
        - 0.005324020628287001 * logtev**6
        + 0.0001975705312221 * logtev**7
        - 3.165581065665e-6 * logtev**8
    )
    vals.at[_i2].set(tiny)
    return vals


# -- k06 --
# @reaction('k06', [(1,HeIII),   (1,de)], [ (1,HeII),         ])
def rxn_k06(tev, tK, logtev):
    vals = 3.36e-10 / jnp.sqrt(tK) / (tK / 1.0e3) ** 0.2 / (1 + (tK / 1.0e6) ** 0.7)
    return vals


# -- k07 --
# @reaction('k07', [   (1,HI),   (1,de)], [   (1,HM),         ])
def rxn_k07(tev, tK, logtev):
    vals = 6.77e-15 * tev**0.8779
    return vals


# -- k08 --
# @reaction('k08', [   (1,HM),   (1,HI)], [  (1,H2I),   (1,de)])
def rxn_k08(tev, tK, logtev):
    _i1 = tev > 0.1
    _i2 = ~_i1
    vals = jnp.exp(
        -20.06913897587003
        + 0.2289800603272916 * logtev
        + 0.03599837721023835 * logtev**2
        - 0.004555120027032095 * logtev**3
        - 0.0003105115447124016 * logtev**4
        + 0.0001073294010367247 * logtev**5
        - 8.36671960467864e-6 * logtev**6
        + 2.238306228891639e-7 * logtev**7
    )
    vals.at[_i2].set(1.43e-9)
    return vals


# -- k09 --


# @reaction('k09', [   (1,HI),  (1,HII)], [ (1,H2II),         ])
def rxn_k09(tev, tK, logtev):
    _i1 = tK > 6.7e3
    vals = 1.85e-23 * tK**1.8
    vals.at[_i1].set(
        5.81e-16 * (tK[_i1] / 56200) ** (-0.6657 * jnp.log10(tK[_i1] / 56200))
    )
    return vals


# -- k10 --
# @reaction('k10', [ (1,H2II),   (1,HI)], [  (1,H2I),  (1,HII)])
def rxn_k10(tev, tK, logtev):
    vals = tK * 0.0 + 6.0e-10
    return vals


# -- k11 --
# @reaction('k11', [  (1,H2I),  (1,HII)], [  (1,H2II),  (1,HI)])
def rxn_k11(tev, tK, logtev):
    _i1 = tev > 0.3
    _i2 = ~_i1
    vals = jnp.exp(
        -24.24914687731536
        + 3.400824447095291 * logtev
        - 3.898003964650152 * logtev**2
        + 2.045587822403071 * logtev**3
        - 0.5416182856220388 * logtev**4
        + 0.0841077503763412 * logtev**5
        - 0.007879026154483455 * logtev**6
        + 0.0004138398421504563 * logtev**7
        - 9.36345888928611e-6 * logtev**8
    )
    vals.at[_i2].set(tiny)
    return vals


# -- k12 --
# @reaction('k12', [  (1,H2I),   (1,de)], [  (2,HII),   (1,de)])
def rxn_k12(tev, tK, logtev):
    _i1 = tev > 0.3
    _i2 = ~_i1
    vals = 5.6e-11 * jnp.exp(-102124 / tK) * tK**0.5
    return vals


# -- k13 --
# NOTE: This is the Glover 2008 rate
# @reaction('k13', [  (1,H2I),   (1,HI)], [   (3,HI),         ])
def rxn_k13(tev, tK, logtev):
    vals = 10.0 ** (
        -178.4239
        - 68.42243 * jnp.log10(tK)
        + 43.20243 * jnp.log10(tK) ** 2
        - 4.633167 * jnp.log10(tK) ** 3
        + 69.70086 * jnp.log10(1 + 40870.38 / tK)
        - (23705.7 / tK)
    )
    return vals


# -- k14 --
# @reaction('k14', [   (1,HM),   (1,de)], [   (1,HI),   (2,de)])
def rxn_k14(tev, tK, logtev):
    _i1 = tev > 0.04
    _i2 = ~_i1
    vals = jnp.exp(
        -18.01849334273
        + 2.360852208681 * logtev
        - 0.2827443061704 * logtev**2
        + 0.01623316639567 * logtev**3
        - 0.03365012031362999 * logtev**4
        + 0.01178329782711 * logtev**5
        - 0.001656194699504 * logtev**6
        + 0.0001068275202678 * logtev**7
        - 2.631285809207e-6 * logtev**8
    )
    vals.at[_i2].set(tiny)
    return vals


# -- k15 --
# @reaction('k15', [   (1,HM),   (1,HI)], [   (2,HI),   (1,de)])
def rxn_k15(tev, tK, logtev):
    _i1 = tev > 0.1
    _i2 = ~_i1
    vals = jnp.exp(
        -20.37260896533324
        + 1.139449335841631 * logtev
        - 0.1421013521554148 * logtev**2
        + 0.00846445538663 * logtev**3
        - 0.0014327641212992 * logtev**4
        + 0.0002012250284791 * logtev**5
        + 0.0000866396324309 * logtev**6
        - 0.00002585009680264 * logtev**7
        + 2.4555011970392e-6 * logtev**8
        - 8.06838246118e-8 * logtev**9
    )
    vals.at[_i2].set(2.56e-9 * tev[_i2] ** 1.78186)
    return vals


# -- k16 --
# @reaction('k16', [   (1,HM),  (1,HII)], [   (2,HI),         ])
def rxn_k16(tev, tK, logtev):
    k16 = 6.5e-9 / jnp.sqrt(tev)
    return k16


# -- k17 --
# @reaction('k17', [   (1,HM),  (1,HII)], [ (1,H2II),   (1,de)])
def rxn_k17(tev, tK, logtev):
    _i1 = tK < 1e4
    _i2 = ~_i1
    vals = 1.0e-8 * tK ** (-0.4)
    vals.at[_i2].set(4.0e-4 * tK[_i2] ** (-1.4) * jnp.exp(-15100.0 / tK[_i2]))
    return vals


# -- k18 --
# @reaction('k18', [ (1,H2II),   (1,de)], [   (2,HI),         ])
def rxn_k18(tev, tK, logtev):
    _i1 = tK > 617
    _i2 = ~_i1
    vals = 1.32e-6 * tK ** (-0.76)
    vals.at[_i2].set(1.0e-8)
    return vals


# -- k19 --
# @reaction('k19', [ (1,H2II),   (1,HM)], [   (1,HI),  (1,H2I)])
def rxn_k19(tev, tK, logtev):
    vals = 5.0e-7 * jnp.sqrt(100.0 / tK)
    return vals


# -- k21 --
# @reaction('k21', [   (2,HI),  (1,H2I)], [  (2,H2I),         ])
def rxn_k21(tev, tK, logtev):
    vals = 2.8e-31 * (tK ** (-0.6))
    return vals


# -- k22 --
# NOTE: This is the Glover 2008 rate
# @reaction('k22', [   (2,HI),   (1,HI)], [  (1,H2I),   (1,HI)])
def rxn_k22(tev, tK, logtev):
    vals = 7.7e-31 / tK**0.464
    return vals


# -- k23 --
# @reaction('k23', [  (1,H2I),  (1,H2I)], [   (2,HI),  (1,H2I)])
def rxn_k23(tev, tK, logtev):
    vals = (
        (8.125e-8 / jnp.sqrt(tK)) * jnp.exp(-52000 / tK) * (1.0 - jnp.exp(-6000 / tK))
    )
    return vals
