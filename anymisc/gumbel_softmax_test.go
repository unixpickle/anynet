package anymisc

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestGumbelSoftmax(t *testing.T) {
	c := anyvec32.CurrentCreator()
	vec := c.MakeVectorData(c.MakeNumericList([]float64{
		1.306852819440055,
		0.796027195674064,
		0.390562087565900,
	}))
	maxHistogram := map[int]int{}
	layer := &GumbelSoftmax{Temperature: 1}
	numIters := 10000
	for i := 0; i < numIters; i++ {
		out := layer.Apply(anydiff.NewConst(vec), 1).Output()
		maxHistogram[anyvec.MaxIndex(out)]++
	}
	expectedFracs := []float64{0.5, 0.3, 0.2}
	for idx, count := range maxHistogram {
		frac := float64(count) / float64(numIters)
		expected := expectedFracs[idx]
		if math.Abs(frac-expected) > 1e-2 {
			t.Error("bad histogram:", maxHistogram)
			break
		}
	}
}
