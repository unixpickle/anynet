package anysgd

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

type testSample struct {
	X2 float64
	Y2 float64
	XY float64
	X  float64
	Y  float64
}

func (t *testSample) Apply(x, y anydiff.Res) anydiff.Res {
	mk := x.Output().Creator().MakeNumeric
	a := anydiff.Scale(anydiff.Mul(x, x), mk(t.X2))
	b := anydiff.Scale(anydiff.Mul(y, y), mk(t.Y2))
	c := anydiff.Scale(anydiff.Mul(x, y), mk(t.XY))
	d := anydiff.Scale(x, mk(t.X))
	e := anydiff.Scale(y, mk(t.Y))
	return anydiff.Add(
		anydiff.Add(a, b),
		anydiff.Add(anydiff.Add(c, d), e),
	)
}

type testSampleList []*testSample

func newTestSampleList() testSampleList {
	// Together, these polynomials add up to 3x^2+3xy-2x+y^2.
	// The global minimum is (x = 4/3, y = -2).
	return testSampleList{
		{X2: 2, X: -1, XY: 0, Y2: 0.5},
		{X2: -1, X: 0, XY: 2, Y2: 0.5},
		{X2: 2, X: -1, XY: 1, Y2: 0},
	}
}

func (t testSampleList) Len() int {
	return len(t)
}

func (t testSampleList) Swap(i, j int) {
	t[i], t[j] = t[j], t[i]
}

func (t testSampleList) Slice(i, j int) SampleList {
	return append(testSampleList{}, t[i:j]...)
}

type testStopper struct {
	callsRemaining int
}

func (t *testStopper) Done() bool {
	t.callsRemaining--
	return t.callsRemaining < 0
}

type testGradienter struct {
	X *anydiff.Var
	Y *anydiff.Var
}

func newTestGradienter(c anyvec.Creator) *testGradienter {
	return &testGradienter{
		X: anydiff.NewVar(c.MakeVector(1)),
		Y: anydiff.NewVar(c.MakeVector(1)),
	}
}

func (t *testGradienter) Gradient(s SampleList) anydiff.Grad {
	var cost anydiff.Res
	for _, x := range s.(testSampleList) {
		res := x.Apply(t.X, t.Y)
		if cost == nil {
			cost = res
		} else {
			cost = anydiff.Add(cost, res)
		}
	}
	grad := anydiff.Grad{
		t.X: t.X.Vector.Creator().MakeVector(1),
		t.Y: t.Y.Vector.Creator().MakeVector(1),
	}
	oneVec := t.X.Vector.Creator().MakeVectorData(
		t.X.Vector.Creator().MakeNumericList([]float64{1}),
	)
	cost.Propagate(oneVec, grad)
	return grad
}

func TestSGD(t *testing.T) {
	g := newTestGradienter(anyvec32.DefaultCreator{})
	s := &SGD{
		Gradienter: g,
		Samples:    newTestSampleList(),
		Rater:      ConstRater(0.0002),
		BatchSize:  1,
	}

	s.Run(&testStopper{callsRemaining: 400000})

	x := g.X.Vector.Data().([]float32)[0]
	y := g.Y.Vector.Data().([]float32)[0]
	if math.Abs(float64(x)-4.0/3) > 1e-2 {
		t.Errorf("bad x value: %f", x)
	}
	if math.Abs(float64(y)+2) > 1e-2 {
		t.Errorf("bad y value: %f", y)
	}
}
