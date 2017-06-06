package anysgd

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestAdamValues(t *testing.T) {
	v := anydiff.NewVar(anyvec32.MakeVector(2))
	g := anydiff.Grad{v: anyvec32.MakeVector(2)}

	a := Adam{DecayRate1: 0.9, DecayRate2: 0.99, Damping: 1e-8}

	scaling1 := math.Sqrt(1-0.99) / (1 - 0.9)
	scaling2 := math.Sqrt(1-math.Pow(0.99, 2)) / (1 - math.Pow(0.9, 2))

	inputGrads := [][]float32{{1, -2}, {2, 1}}
	expectedOuts := [][]float64{
		{
			0.1 * scaling1 / math.Sqrt(0.01+1e-8),
			0.1 * -2 * scaling1 / math.Sqrt(0.04+1e-8),
		},
		{
			scaling2 * (0.9*0.1 + 0.1*2) / math.Sqrt((0.99*0.01+0.01*4)+1e-8),
			scaling2 * (-0.9*0.2 + 0.1) / math.Sqrt((0.99*0.04+0.01)+1e-8),
		},
	}

	for i, input := range inputGrads {
		g[v].SetData(input)
		actual := a.Transform(g)[v].Data().([]float32)
		expected := expectedOuts[i]
		for j, x := range expected {
			act := actual[j]
			if math.IsNaN(float64(act)) || math.IsNaN(x) ||
				math.Abs(float64(act)-x) > 1e-3 {
				t.Errorf("time %d out %d: expected %f, got %f", i, j, x, act)
			}
		}
	}
}

func TestAdamTraining(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	stop := newTestStopper(100000)
	g := newTestGradienter()
	s := &SGD{
		Fetcher:     testFetcher{},
		Gradienter:  g,
		Transformer: &Adam{},
		Samples:     newTestSampleList(),
		Rater:       ConstRater(0.001),
		StatusFunc:  stop.StatusFunc,
		BatchSize:   1,
	}

	s.Run(stop.Chan())

	if g.errorMargin() > 1e-2 {
		x, y := g.current()
		t.Errorf("bad solution: %f, %f", x, y)
	}
}

func TestAdamMarshal(t *testing.T) {
	c := anyvec64.DefaultCreator{}
	a := &Adam{
		DecayRate1: 0.3,
		DecayRate2: 0.4,
		Damping:    0.2,
		Vars:       randomVars(c),
	}
	testMarshal(t, a, a.Vars)
}
