package anyctc

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestCostOutputs(t *testing.T) {
	probSeqs := [][][]float64{
		{},
		{{0.3, 0.2, 0.5}, {0.1, 0.5, 0.4}},
		{},
	}
	labels := [][]int{{}, {0, 1}, {1}}
	expectedProbs := []float64{1, 0.3 * 0.5, 0}
	inSeqs := logProbSeqs(anyvec32.CurrentCreator(), probSeqs)
	negOut := anydiff.Scale(Cost(inSeqs, labels), float32(-1))
	actualProbs := vectorFloats(anydiff.Exp(negOut).Output())
	for i, x := range expectedProbs {
		a := actualProbs[i]
		if math.Abs(x-a)/x > testPrecision {
			t.Errorf("output %d: expected %f but got %f", i, x, a)
		}
	}
}

func TestCostGrad(t *testing.T) {
	c := anyvec32.CurrentCreator()
	var vars []*anydiff.Var
	seqs := anyseq.ResSeq(c, []*anyseq.ResBatch{
		{Packed: randomVar(c, 9, &vars), Present: []bool{true, true, true}},
		{Packed: randomVar(c, 6, &vars), Present: []bool{true, false, true}},
		{Packed: randomVar(c, 3, &vars), Present: []bool{false, false, true}},
		{Packed: randomVar(c, 3, &vars), Present: []bool{false, false, true}},
	})
	labels := [][]int{{1, 0}, {0}, {0, 1, 2}}
	ch := anydifftest.ResChecker{
		F: func() anydiff.Res {
			return Cost(seqs, labels)
		},
		V:     vars,
		Prec:  testPrecision * 3,
		Delta: testPrecision,
	}
	ch.FullCheck(t)
}

func logProbSeqs(c anyvec.Creator, values [][][]float64) anyseq.Seq {
	vecLists := make([][]anyvec.Vector, len(values))
	for i, seq := range values {
		vecLists[i] = make([]anyvec.Vector, len(seq))
		for j, x := range seq {
			vecLists[i][j] = c.MakeVectorData(c.MakeNumericList(x))
			anyvec.Log(vecLists[i][j])
		}
	}
	return anyseq.ConstSeqList(c, vecLists)
}

func randomVar(c anyvec.Creator, n int, vs *[]*anydiff.Var) *anydiff.Var {
	v := c.MakeVector(n)
	anyvec.Rand(v, anyvec.Normal, nil)
	anyvec.LogSoftmax(v, 3)
	res := anydiff.NewVar(v)
	*vs = append(*vs, res)
	return res
}
