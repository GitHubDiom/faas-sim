from sim.stats import ParameterizedDistribution as PDist

execution_time_distributions = {
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37'): (0.584, 1.1420000000000001, PDist.lognorm(((0.31449780857108944,), 0.36909628354997315, 0.41583220283981315))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37'): (0.434, 0.491, PDist.lognorm(((0.4040891723912467,), 0.42388853817361616, 0.026861281861394234))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37'): (23.89, 108.743, PDist.lognorm(((0.20506023311489027,), -20.283277542855338, 73.25412435048207))),
    ('nuc', 'alexrashed/ml-wf-1-pre:0.37'): (0.10099999999999999, 0.105, PDist.lognorm(((23.561803013427763,), 0.10099999999999998, 0.00012383535141814733))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37'): (156.256, 166.702, PDist.lognorm(((0.0955189827829152,), 140.38073254297746, 20.5799433435967))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37'): (17.641, 20.333, PDist.lognorm(((0.07553515034079655,), 12.501696816822736, 6.339535271582099))),
    ('nuc', 'alexrashed/ml-wf-2-train:0.37'): (31.326999999999998, 42.355, PDist.lognorm(((0.08117635531817702,), 4.01869140845729, 32.19443389130219))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37'): (0.542, 11.505, PDist.lognorm(((2.148861851180003,), 0.5402853007866957, 1.422846600303037))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37'): (0.48, 10.882, PDist.lognorm(((4.521620371690464,), 0.4799999929591894, 5.3308588402679185))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37'): (1.203, 14.579, PDist.lognorm(((0.0011447901398451941,), -2722.59408910652, 2733.773971903728))),
    ('nuc', 'alexrashed/ml-wf-3-serve:0.37'): (0.14400000000000002, 10.261, PDist.lognorm(((3.0008797752241914,), 0.14399171198249225, 0.0833885862659985))),
}

# distributions were fitted on the 100mb preset
startup_time_distributions = {
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37', True): (4.845974683761597, 8.010702610015871, PDist.lognorm(((0.0037449457378122934,), -184.7438193180742, 191.23179742021864))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37', True): (1.4631223678588867, 2.677159309387207, PDist.lognorm(((0.002512746872799938,), -105.01241065619348, 107.174808474558))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37', True): (2.5531022548675537, 30.92630386352539, PDist.lognorm(((0.8377300856682666,), 1.9047810452600982, 5.02386592606429))),
    ('nuc', 'alexrashed/ml-wf-1-pre:0.37', True): (1.4307518005371094, 2.55591344833374, PDist.lognorm(((0.0010473161054870804,), -220.55445796259818, 222.70478363394423))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37', True): (4.694065570831299, 8.718086004257202, PDist.lognorm(((0.005819346589794688,), -128.07418871497953, 135.10321855444715))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37', True): (1.4736998081207275, 2.805474281311035, PDist.lognorm(((0.0043443267299934,), -64.65543175628802, 66.81621184023626))),
    ('pi', 'alexrashed/ml-wf-2-train:0.37', True): (2.518429517745972, 41.4433479309082, PDist.lognorm(((0.9050857042528122,), 1.9348219443423567, 5.316676365244197))),
    ('nuc', 'alexrashed/ml-wf-2-train:0.37', True): (1.4586496353149414, 2.555532693862915, PDist.lognorm(((0.001681706956053075,), -144.2795731356258, 146.42393882044934))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37', True): (5.1318864822387695, 8.278877973556519, PDist.lognorm(((0.0027484186792601697,), -165.77166614636923, 172.71014758735464))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37', True): (1.5105795860290527, 2.8387768268585205, PDist.lognorm(((0.00897732092441757,), -27.91705469284816, 30.07755167658162))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37', True): (2.4959282875061035, 24.689355611801147, PDist.lognorm(((0.7194917986461948,), 1.7939233642283035, 3.69032959681382))),
    ('nuc', 'alexrashed/ml-wf-3-serve:0.37', True): (1.438429594039917, 3.182549238204956, PDist.lognorm(((0.002914306587106721,), -93.2468082577646, 95.48311762857361))),
    ('cloud', 'alexrashed/ml-wf-1-pre:0.37', False): (97.12672901153564, 824.8937194347383, PDist.lognorm(((4.859615995909446,), 97.12672901147204, 1.14796454650864))),
    ('tegra', 'alexrashed/ml-wf-1-pre:0.37', False): (58.74252128601074, 750.2965703010559, PDist.lognorm(((4.039162429062674,), 58.742520450944646, 10.172277144411156))),
    ('pi', 'alexrashed/ml-wf-1-pre:0.37', False): (673.4305050373077, 1628.758401632309, PDist.lognorm(((6.447150531564345,), 673.4305050373076, 1.4302943416705516))),
    ('nuc', 'alexrashed/ml-wf-1-pre:0.37', False): (21.203427076339718, 762.7251296043396, PDist.lognorm(((3.088813855827093,), 21.203353546686493, 1.2407368468763815))),
    ('cloud', 'alexrashed/ml-wf-2-train:0.37', False): (97.64930057525636, 815.3668050765991, PDist.lognorm(((1.9077194079252626,), 97.33879731144327, 98.36218082195288))),
    ('tegra', 'alexrashed/ml-wf-2-train:0.37', False): (59.60635018348694, 787.383499622345, PDist.lognorm(((1.6586371453566358,), 57.85331573109663, 131.8688446326562))),
    ('pi', 'alexrashed/ml-wf-2-train:0.37', False): (775.337126493454, 1921.5559482574465, PDist.lognorm(((5.265736296171748,), 775.3371264934526, 1.552558429806739))),
    ('nuc', 'alexrashed/ml-wf-2-train:0.37', False): (21.228580951690677, 762.246561050415, PDist.lognorm(((2.3800082435142587,), 21.1771924016278, 158.16980950244087))),
    ('cloud', 'alexrashed/ml-wf-3-serve:0.37', False): (103.02121186256409, 835.4177339076996, PDist.lognorm(((4.03394744724086,), 103.02121157275369, 3.386806662755468))),
    ('tegra', 'alexrashed/ml-wf-3-serve:0.37', False): (60.83073115348816, 796.7554597854614, PDist.lognorm(((1.3468144640330468,), 52.974382739563765, 163.116500342865))),
    ('pi', 'alexrashed/ml-wf-3-serve:0.37', False): (833.7670969963074, 1974.2024960517886, PDist.lognorm(((6.243612659761245,), 833.7670969963071, 1.488051925178726))),
    ('nuc', 'alexrashed/ml-wf-3-serve:0.37', False): (22.16527533531189, 794.675945520401, PDist.lognorm(((2.4095393620098946,), 22.08610367498848, 80.78142071894206))),

}
