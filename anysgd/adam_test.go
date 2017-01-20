package anysgd

import "testing"

func TestAdam(t *testing.T) {
	g := newTestGradienter()
	s := &SGD{
		Gradienter:  g,
		Transformer: &Adam{},
		Samples:     newTestSampleList(),
		Rater:       ConstRater(0.001),
		BatchSize:   1,
	}

	s.Run(&testStopper{callsRemaining: 100000})

	if g.errorMargin() > 1e-2 {
		x, y := g.current()
		t.Errorf("bad solution: %f, %f", x, y)
	}
}
