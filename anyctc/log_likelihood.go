package anyctc

import (
	"math"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// logLikelihood computes the log likelihood of the label.
// The last entry of each input vector is the log of
// the probability of the blank symbol.
//
// This only works for creators that use []float64 numeric
// list types.
func logLikelihood(c anyvec.Creator, seq []anydiff.Res, label []int) anydiff.Res {
	if len(seq) == 0 {
		var constVal float64
		if len(label) == 0 {
			constVal = 0
		} else {
			constVal = math.Inf(-1)
		}
		numList := c.MakeNumericList([]float64{constVal})
		return anydiff.NewVar(c.MakeVectorData(numList))
	}

	// Don't want the result to retain a reference to
	// this slice.
	label = append([]int{}, label...)

	// positionProbs stores the log probabilities of
	// being at every position in the blank-infused
	// label, where blanks are injected at the start
	// and end of the label, and between entries.
	var positionProbs anydiff.Res

	initProbs := make([]float64, len(label)*2+1)
	for i := 1; i < len(initProbs); i++ {
		initProbs[i] = math.Inf(-1)
	}
	positionProbs = anydiff.NewConst(c.MakeVectorData(c.MakeNumericList(initProbs)))

	for _, inRes := range seq {
		positionProbs = newLogLikelihoodStep(inRes, positionProbs, label)
	}

	// May occur if the label is empty.
	if positionProbs.Output().Len() == 1 {
		return anydiff.Slice(positionProbs, 0, 1)
	}

	return addLastTwoLogs(positionProbs)
}

type logLikelihoodStep struct {
	OutVec    anyvec.Vector
	LastProbs anydiff.Res
	SeqIn     anydiff.Res
	Label     []int
	V         anydiff.VarSet
}

func newLogLikelihoodStep(inputRes, positionProbs anydiff.Res, label []int) *logLikelihoodStep {
	input := inputRes.Output().Data().([]float64)
	last := positionProbs.Output().Data().([]float64)
	newProbs := make([]float64, len(last))
	newProbs[0] = last[0] + input[len(input)-1]
	for i := 2; i < len(label)*2+1; i += 2 {
		newProbs[i] = addLogs(last[i-1], last[i]) + input[len(input)-1]
	}
	for i := 1; i < len(label)*2+1; i += 2 {
		labelIdx := (i - 1) / 2
		positionSum := addLogs(last[i], last[i-1])
		if labelIdx > 0 && label[labelIdx-1] != label[labelIdx] {
			positionSum = addLogs(positionSum, last[i-2])
		}
		newProbs[i] = input[label[labelIdx]] + positionSum
	}
	c := inputRes.Output().Creator()
	return &logLikelihoodStep{
		OutVec:    c.MakeVectorData(c.MakeNumericList(newProbs)),
		LastProbs: positionProbs,
		SeqIn:     inputRes,
		Label:     label,
		V:         anydiff.MergeVarSets(inputRes.Vars(), positionProbs.Vars()),
	}
}

func (l *logLikelihoodStep) Output() anyvec.Vector {
	return l.OutVec
}

func (l *logLikelihoodStep) Vars() anydiff.VarSet {
	return l.V
}

func (l *logLikelihoodStep) Propagate(u anyvec.Vector, g anydiff.Grad) {
	upstream := u.Data().([]float64)
	last := l.LastProbs.Output().Data().([]float64)
	input := l.SeqIn.Output().Data().([]float64)
	lastGrad := make([]float64, len(last))
	inputGrad := make([]float64, len(input))

	lastGrad[0] = upstream[0]
	inputGrad[len(inputGrad)-1] = upstream[0]

	for i := 2; i < len(l.Label)*2+1; i += 2 {
		inputGrad[len(inputGrad)-1] += upstream[i]
		da, db := addLogsDeriv(last[i-1], last[i], upstream[i])
		lastGrad[i-1] += da
		lastGrad[i] += db
	}
	for i := 1; i < len(l.Label)*2+1; i += 2 {
		labelIdx := (i - 1) / 2
		inputGrad[l.Label[labelIdx]] += upstream[i]
		if labelIdx > 0 && l.Label[labelIdx-1] != l.Label[labelIdx] {
			a := addLogs(last[i-2], last[i-1])
			b := last[i]
			da, db := addLogsDeriv(a, b, upstream[i])
			lastGrad[i] += db
			da, db = addLogsDeriv(last[i-2], last[i-1], da)
			lastGrad[i-2] += da
			lastGrad[i-1] += db
		} else {
			da, db := addLogsDeriv(last[i-1], last[i], upstream[i])
			lastGrad[i-1] += da
			lastGrad[i] += db
		}
	}

	c := l.LastProbs.Output().Creator()
	if g.Intersects(l.LastProbs.Vars()) {
		l.LastProbs.Propagate(c.MakeVectorData(lastGrad), g)
	}
	if g.Intersects(l.SeqIn.Vars()) {
		l.SeqIn.Propagate(c.MakeVectorData(inputGrad), g)
	}
}

// addLogs adds two numbers in the log domain.
func addLogs(a, b float64) float64 {
	if math.IsInf(a, -1) {
		return b
	} else if math.IsInf(b, -1) {
		return a
	}
	normalizer := math.Max(a, b)
	exp1 := math.Exp(a - normalizer)
	exp2 := math.Exp(b - normalizer)
	return math.Log(exp1+exp2) + normalizer
}

// addLogsDeriv computes the partial derivatives for
// addition in the log domain.
func addLogsDeriv(a, b, upstream float64) (da, db float64) {
	if math.IsInf(a, -1) && math.IsInf(b, -1) {
		return
	}
	denomLog := addLogs(a, b)
	daLog := a - denomLog
	dbLog := b - denomLog
	da = upstream * math.Exp(daLog)
	db = upstream * math.Exp(dbLog)
	return
}

type addLastTwoLogsRes struct {
	In     anydiff.Res
	OutVec anyvec.Vector
}

func addLastTwoLogs(res anydiff.Res) anydiff.Res {
	v := res.Output().Data().([]float64)
	sum := res.Output().Creator().MakeNumericList([]float64{
		addLogs(v[len(v)-1], v[len(v)-2]),
	})
	return &addLastTwoLogsRes{
		In:     res,
		OutVec: res.Output().Creator().MakeVectorData(sum),
	}
}

func (a *addLastTwoLogsRes) Output() anyvec.Vector {
	return a.OutVec
}

func (a *addLastTwoLogsRes) Vars() anydiff.VarSet {
	return a.In.Vars()
}

func (a *addLastTwoLogsRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	v := a.In.Output().Data().([]float64)
	da, db := addLogsDeriv(v[len(v)-1], v[len(v)-2], u.Data().([]float64)[0])
	downstream := make([]float64, len(v))
	downstream[len(v)-1] = da
	downstream[len(v)-2] = db
	a.In.Propagate(u.Creator().MakeVectorData(downstream), g)
}
