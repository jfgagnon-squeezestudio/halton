import matplotlib.pyplot as plt
import numpy as np


def haltonRefImpl(i, base, scrambleObject):
    invFloatBase = 1.0 / float(base)

    value = 0.0
    divisor = invFloatBase

    if (scrambleObject is not None):
        scrambleObject.init(i, base)

    while((1.0 - divisor) < 1.0):
        digit = i % base

        if (scrambleObject is not None):
            digit = scrambleObject.apply(digit)

        value += digit * divisor

        divisor *= invFloatBase
        i = (i - digit) // base

    return value


def haltonOptimized(i, base, scrambleObject):
    """
    Not yet fully done.
    """
    invFloatBase = 1.0 / float(base)

    value = 0.0
    divisor = invFloatBase

    if (scrambleObject is not None):
        scrambleObject.init(i, base)

    while(i != 0):
        digit = i % base

        if (scrambleObject is not None):
            valueDigit = scrambleObject.apply(digit)
        else:
            valueDigit = digit

        value += valueDigit * divisor

        divisor *= invFloatBase
        i = i // base

    return value


class ScramblePerSample:
    def __init__(self, base):
        pass

    def init(self, i, base):
        self._digitsPermutation = np.arange(0, base, 1, dtype=np.uint32)
        np.random.shuffle(self._digitsPermutation)

    def apply(self, digit):
        return self._digitsPermutation[digit]

    @classmethod
    def resetSeed(cls):
        np.random.seed(0)


class ScramblePerBase:
    def __init__(self, base):
        self._digitsPermutation = np.arange(0, base, 1, dtype=np.uint32)
        np.random.shuffle(self._digitsPermutation)

    def init(self, i, base):
        pass

    def apply(self, digit):
        return self._digitsPermutation[digit]

    @classmethod
    def resetSeed(cls):
        np.random.seed(0)


def generateHalton2D(sequenceStart, sequenceStop, sequenceStep, base0, base1, scrambleObject0, scrambleObject1, haltonFunc):
    halton2D = np.vectorize(lambda i: (haltonFunc(i, base0, scrambleObject0), haltonFunc(i, base1, scrambleObject1)))
    sequence = np.arange(sequenceStart, sequenceStop, sequenceStep, dtype=np.uint32)
    return halton2D(sequence)


def plotHalton2D(subplot, start, stop, base0, base1, scrambleClass, haltonFunc, step=1):
    if scrambleClass is None:
        scramble0 = None
        scramble1 = None
    else:
        # reset seed so we always get same sequence
        scrambleClass.resetSeed()
        scramble0 = scrambleClass(base0)
        scramble1 = scrambleClass(base1)
    sequence = generateHalton2D(start,
                                stop,
                                step,
                                base0,
                                base1,
                                scramble0,
                                scramble1,
                                haltonFunc)
    scrambleName = "no scrambling" if scrambleClass is None else scrambleClass.__name__
    subplot.set_title(f"{haltonFunc.__name__} ({base0}, {base1}) - {scrambleName}")
    subplot.set_xlim([0.0, 1.0])
    subplot.set_ylim([0.0, 1.0])
    subplot.plot(sequence[0], sequence[1], '.')
    subplot.grid()


def generateRandom(sequenceStart, sequenceStop):
    # reset seed so we always get same sequence
    np.random.seed(0)
    random = np.vectorize(lambda i: (np.random.random(), np.random.random()))
    sequence = np.arange(sequenceStart, sequenceStop, 1, dtype=np.uint32)
    return random(sequence)


# -----------------------------


def doScrambleStudy(start, stop, base0, base1, base2, base3):
    """
    Shows impact of scrambling for higher dimension sequence. Conclusion is that for
    low dimension, scramble does not give much but is mandatory for higher dimensions.
    """
    # try different halton implementation
    haltonFunc = haltonRefImpl

    # plot each results
    fig, ax = plt.subplots(4, 2, figsize=(15, 12), tight_layout=True)
    fig.suptitle(f"Halton2D scramble study - {stop - start} samples")

    # generate data in different manner
    # complete random data
    rand = generateRandom(start, stop)
    ax[0][0].set_title('Random')
    ax[0][0].set_xlim([0.0, 1.0])
    ax[0][0].set_ylim([0.0, 1.0])
    ax[0][0].plot(rand[0], rand[1], 'g.')
    ax[0][0].grid()

    # halton2D vanilla
    plotHalton2D(ax[1][0], start, stop, base0, base1, None, haltonFunc)
    plotHalton2D(ax[1][1], start, stop, base2, base3, None, haltonFunc)

    # halton2D scramble per sample
    plotHalton2D(ax[2][0], start, stop, base0, base1, ScramblePerSample, haltonFunc)
    plotHalton2D(ax[2][1], start, stop, base2, base3, ScramblePerSample, haltonFunc)

    # halton2D scramble per base
    plotHalton2D(ax[3][0], start, stop, base0, base1, ScramblePerBase, haltonFunc)
    plotHalton2D(ax[3][1], start, stop, base2, base3, ScramblePerBase, haltonFunc)

    # compare distribution
    plt.savefig("halton2d_scramble_study.png")


def doImplStudy(start, stop, base0, base1, base2, base3):
    """
    Shows impact of sequence implementation. Mostly impact scrambling.
    Goal is to find an efficient GPU version given that random number are not
    that easy to get on GPU.

    - First test was to get rid of (1.0 - divisor) < 1.0 test. This cause
      iteration to go for a long time but it seems necessary for higher
      precision. Low dimensions do not suffer much though.

    - Further tests are needed to optimize scrambling.
    """
    # plot each results
    fig, ax = plt.subplots(4, 2, figsize=(15, 12), tight_layout=True)
    fig.suptitle(f"Halton2D impl study - {stop - start} samples")

    # generate data in different manner
    # complete random data
    rand = generateRandom(start, stop)
    ax[0][0].set_title('Random')
    ax[0][0].set_xlim([0.0, 1.0])
    ax[0][0].set_ylim([0.0, 1.0])
    ax[0][0].plot(rand[0], rand[1], 'g.')
    ax[0][0].grid()

    # halton2D vanilla
    plotHalton2D(ax[1][0], start, stop, base0, base1, None, haltonRefImpl)
    plotHalton2D(ax[1][1], start, stop, base2, base3, None, haltonRefImpl)

    # halton2D - ref vs optimized loop
    plotHalton2D(ax[2][0], start, stop, base0, base1, ScramblePerBase, haltonRefImpl)
    plotHalton2D(ax[2][1], start, stop, base2, base3, ScramblePerBase, haltonRefImpl)
    plotHalton2D(ax[3][0], start, stop, base0, base1, ScramblePerBase, haltonOptimized)
    plotHalton2D(ax[3][1], start, stop, base2, base3, ScramblePerBase, haltonOptimized)

    # compare distribution
    plt.savefig("halton2d_impl_study.png")


def doSequenceStudy(start, stop, base0, base1):
    """
    Show impact of using sub sequence. Look for colored point distribution.
    """
    numSequence = 4

    # plot each results
    fig, ax = plt.subplots(4, numSequence, figsize=(15, 12), tight_layout=True)
    fig.suptitle(f"Halton2D sub sequence study - {stop - start} samples")

    # halton2D vanilla
    plotHalton2D(ax[0][0], start, stop, base0, base1, ScramblePerBase, haltonOptimized)

    # halton2D - cut sequence
    seqSize = (start + stop) // numSequence

    # consecutive sub sequence
    # put everything in 1 graph to see global distribution
    startCopy = start
    while (startCopy < stop):
        plotHalton2D(ax[1][0], startCopy, startCopy + seqSize, base0, base1, ScramblePerBase, haltonOptimized)
        startCopy += seqSize

    # consecutive sub sequence
    # put sub sequence in a different plot to see each distribution
    i = 0
    startCopy = start
    while (startCopy < stop):
        plotHalton2D(ax[2][i], startCopy, startCopy + seqSize, base0, base1, ScramblePerBase, haltonOptimized)
        startCopy += seqSize
        i += 1

    # every other numSequence sub sequence
    for i in range(numSequence):
        plotHalton2D(ax[3][i], start + i, stop - numSequence + i + 1, base0, base1, ScramblePerBase, haltonOptimized, numSequence)

    # compare distribution
    plt.savefig("halton2d_sequence_study.png")


def doSequenceStartStudy(start, stop, base0, base1):
    """
    Show impact of start point on consecutive sequence.
    """
    numSequence = 4

    size = stop - start
    iteration = [(0, 0), (2 * size, 1), (5 * size, 2)]

    # plot each results
    fig, ax = plt.subplots(5, numSequence, figsize=(15, 12), tight_layout=True)
    fig.suptitle(f"Halton2D sub sequence start study - {stop - start} samples")

    # halton2D vanilla with start offset
    for offset, i in iteration:
        plotHalton2D(ax[0][i], start + offset, stop + offset, base0, base1, ScramblePerBase, haltonOptimized)

    # halton2D - cut sequence
    seqSize = (start + stop) // numSequence

    # consecutive sub sequence with start offset
    # put everything in 1 graph to see global distribution
    for offset, i in iteration:
        startCopy = start
        while (startCopy < stop):
            plotHalton2D(ax[1][i], startCopy + offset, startCopy + seqSize + offset, base0, base1, ScramblePerBase, haltonOptimized)
            startCopy += seqSize

    # consecutive sub sequence with start offset
    # put sub sequence in a different plot to see each distribution
    for offset, j in iteration:
        i = 0
        startCopy = start
        while (startCopy < stop):
            plotHalton2D(ax[2 + j][i], startCopy + offset, startCopy + seqSize + offset, base0, base1, ScramblePerBase, haltonOptimized)
            startCopy += seqSize
            i += 1

    # compare distribution
    plt.savefig("halton2d_sequence_start_study.png")


if __name__ == "__main__":
    # sequence to generate over
    start = 0
    stop = 2000

    # halton2D basis
    base0 = 2; base1 = 3
    base2 = 1103; base3 = 1109
    # base2 = 43; base3 = 47

    # doScrambleStudy(start, stop, base0, base1, base2, base3)
    # plt.clf()

    # doImplStudy(start, stop, base0, base1, base2, base3)
    # plt.clf()

    # doSequenceStudy(start, stop, base0, base1)
    # plt.clf()

    doSequenceStartStudy(start, stop, base0, base1)
    # plt.clf()

    plt.show()
