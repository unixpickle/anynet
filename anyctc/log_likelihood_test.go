package anyctc

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

const (
	testSymbolCount = 5
	testPrecision   = 1e-3
)

func TestLogLikelihoodOutputs(t *testing.T) {
	c := anyvec32.CurrentCreator()
	for i := 0; i < 11; i++ {
		labelLen := 5 + rand.Intn(5)
		if i == 10 {
			labelLen = 0
		}
		seqLen := labelLen + rand.Intn(5)
		label := make([]int, labelLen)
		for i := range label {
			label[i] = rand.Intn(testSymbolCount)
		}
		seq, res := createTestSequence(c, seqLen, testSymbolCount)
		expected := exactLikelihood(seq, label, -1)
		actual := math.Exp(vectorFloats(logLikelihood(c, res, label).Output())[0])
		if math.Abs(actual-expected)/math.Abs(expected) > testPrecision {
			t.Errorf("LogLikelihood gave log(%e) but expected log(%e)",
				actual, expected)
		}
	}
}

func TestLogLikelihoodGrad(t *testing.T) {
	c := anyvec32.CurrentCreator()
	label := make([]int, 5)
	for i := range label {
		label[i] = rand.Intn(testSymbolCount)
	}
	_, resSeq := createTestSequence(c, len(label)+5, testSymbolCount)
	var vars []*anydiff.Var
	for _, x := range resSeq {
		for v := range x.Vars() {
			vars = append(vars, v)
		}
	}
	ch := anydifftest.ResChecker{
		F: func() anydiff.Res {
			return logLikelihood(c, resSeq, label)
		},
		V:     vars,
		Prec:  testPrecision * 2,
		Delta: testPrecision,
	}
	ch.FullCheck(t)
}

// createTestSequence creates a test sequence.
//
// The sequence is produced in two forms.
// First, a native sequence of []float64 containing actual
// probabilities is produced.
// Second, a sequence of anydiff.Res is produced with the
// logs of the probabilities.
func createTestSequence(c anyvec.Creator, seqLen, symCount int) ([][]float64, []anydiff.Res) {
	res := make([]anydiff.Res, seqLen)
	seq := make([][]float64, seqLen)
	for i := range seq {
		seq[i] = make([]float64, symCount+1)
		var probSum float64
		for j := range seq[i] {
			seq[i][j] = math.Abs(rand.NormFloat64())
			probSum += seq[i][j]
		}
		for j := range seq[i] {
			seq[i][j] /= probSum
		}
		logVec := make([]float64, len(seq[i]))
		for j := range logVec {
			logVec[j] = math.Log(seq[i][j])
		}
		res[i] = anydiff.NewVar(c.MakeVectorData(c.MakeNumericList(logVec)))
	}
	return seq, res
}

// exactLikelihood computes the log likelihood of a label
// naively from a sequence of raw, unlogged probabilities.
func exactLikelihood(seq [][]float64, label []int, lastSymbol int) float64 {
	if len(seq) == 0 {
		if len(label) == 0 {
			return 1
		} else {
			return 0
		}
	}

	next := seq[0]
	blank := len(next) - 1

	var res float64
	res += next[blank] * exactLikelihood(seq[1:], label, -1)
	if lastSymbol >= 0 {
		res += next[lastSymbol] * exactLikelihood(seq[1:], label, lastSymbol)
	}
	if len(label) > 0 && label[0] != lastSymbol {
		res += next[label[0]] * exactLikelihood(seq[1:], label[1:], label[0])
	}
	return res
}
