package anynet

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// A Cost provides a way to measure the amount of error
// from the output of a neural network.
//
// Just like regular Layers, a Cost function is batched.
// It takes a packed batch of desired outputs and actual
// outputs, and produces a batch of costs.
type Cost interface {
	Cost(desired, actual anydiff.Res, n int) anydiff.Res
}

// DotCost computes the cost by taking the dot product of
// the desired and actual outputs, and then negating it.
//
// This is meant to be used with LogSoftmax activations.
// When you dot the output of a LogSoftmax with the
// desired probabilities, you get an unbiased measure of
// cross-entropy error.
type DotCost struct{}

// Cost takes the dot product of each actual output with
// each desired output, negates it, and uses that as the
// cost.
func (d DotCost) Cost(desired, actual anydiff.Res, n int) anydiff.Res {
	comb := anydiff.Mul(desired, actual)
	dots := anydiff.SumCols(&anydiff.Matrix{
		Data: comb,
		Rows: n,
		Cols: comb.Output().Len() / n,
	})
	return anydiff.Scale(dots, dots.Output().Creator().MakeNumeric(-1))
}

// MSE evaluates cost as the squared Euclidean distance
// between the actual and desired output.
type MSE struct{}

// Cost computes, for each output, the mean squared
// distance between the actual and desired output value.
func (m MSE) Cost(desired, actual anydiff.Res, n int) anydiff.Res {
	neg := anydiff.Scale(actual, actual.Output().Creator().MakeNumeric(-1))
	diff := anydiff.Add(desired, neg)
	sq := anydiff.Square(diff)
	numComps := sq.Output().Len() / n
	sum := anydiff.SumCols(&anydiff.Matrix{
		Data: sq,
		Rows: n,
		Cols: numComps,
	})
	normalizer := 1.0 / float64(numComps)
	return anydiff.Scale(sum, sum.Output().Creator().MakeNumeric(normalizer))
}

// SigmoidCE combines a sigmoid output activation with
// cross-entropy loss.
type SigmoidCE struct {
	// Average indicates whether or not the cross-entropy
	// cost should be an average rather than a sum.
	Average bool
}

// Cost is mathematically equivalent to applying the
// sigmoid to each component of actual, then finding the
// cross-entropy loss.
func (s SigmoidCE) Cost(desired, actual anydiff.Res, n int) anydiff.Res {
	minusOne := actual.Output().Creator().MakeNumeric(-1)
	costProducts := anydiff.Pool(desired, func(desired anydiff.Res) anydiff.Res {
		return anydiff.Pool(actual, func(actual anydiff.Res) anydiff.Res {
			logRegular := anydiff.LogSigmoid(actual)
			logComplement := anydiff.LogSigmoid(anydiff.Scale(actual, minusOne))
			return anydiff.Add(
				anydiff.Mul(desired, logRegular),
				anydiff.Mul(anydiff.Complement(desired), logComplement),
			)
		})
	})
	res := anydiff.SumCols(&anydiff.Matrix{
		Data: costProducts,
		Rows: n,
		Cols: actual.Output().Len() / n,
	})
	d := -1.0
	if s.Average {
		d /= float64(actual.Output().Len() / n)
	}
	return anydiff.Scale(res, res.Output().Creator().MakeNumeric(d))
}

// L2Reg wraps a Cost and adds an L2 penalty.
//
// The L2 penalty is computed by squaring the parameters,
// summing the squares, then multiplying the sum by
// Penalty / 2.
type L2Reg struct {
	Penalty float64
	Params  []*anydiff.Var
	Wrapped Cost
}

// Cost computes the cost from l.Wrapped and adds the L2
// penalty to each component.
func (l *L2Reg) Cost(desired, actual anydiff.Res, n int) anydiff.Res {
	var sum anydiff.Res
	sum = anydiff.NewConst(actual.Output().Creator().MakeVector(1))
	for _, p := range l.Params {
		sum = anydiff.Add(sum, anydiff.Sum(anydiff.Square(p)))
	}
	sum = anydiff.Scale(sum, sum.Output().Creator().MakeNumeric(l.Penalty/2))
	return anydiff.AddRepeated(l.Wrapped.Cost(desired, actual, n), sum)
}

// Hinge implements binary hinge-loss.
//
// Let d be the desired classification (1 or -1) and let a
// be the actual network output.
// The hinge loss is defined as:
//
//     max(0, 1-a*d)
//
// The loss for each batch is the sum of the hinge loss
// for each component.
type Hinge struct{}

// Cost computes the total hinge loss for each batch.
func (h Hinge) Cost(desired, actual anydiff.Res, n int) anydiff.Res {
	m := &anydiff.Matrix{
		Data: anydiff.ClipPos(anydiff.Complement(anydiff.Mul(desired, actual))),
		Rows: n,
		Cols: desired.Output().Len() / n,
	}
	return anydiff.SumCols(m)
}

// MultiHinge is a multi-class hinge loss.
//
// There are several different ways to generalize the
// hinge loss to multi-class settings.
// The MultiHinge aims to implement those ways.
type MultiHinge int

// CrammerSinger is the hinge loss described in
// http://www.jmlr.org/papers/volume2/crammer01a/crammer01a.pdf and
// http://www.ttic.edu/sigml/symposium2011/papers/Moore+DeNero_Regularization.pdf.
//
// WestonWatkins is the hinge loss described in
// https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es1999-461.pdf and
// http://cs231n.github.io/linear-classify/#svm.
//
// See http://www.jmlr.org/papers/volume17/11-229/11-229.pdf
// for details on different hinge losses.
const (
	CrammerSinger MultiHinge = iota
	WestonWatkins
)

// Cost computes the hinge loss for each batch.
//
// The desired vector for each batch should have all zero
// elements except for a single 1.
// The 1 is taken to mark the spot of the correct class.
//
// Typically, the desired vector should be a constant.
// The gradient of the desired output means nothing, since
// the desired output has a specific binary format.
func (m MultiHinge) Cost(desired, actual anydiff.Res, n int) anydiff.Res {
	return anydiff.Pool(desired, func(desired anydiff.Res) anydiff.Res {
		return anydiff.Pool(actual, func(actual anydiff.Res) anydiff.Res {
			if m == CrammerSinger {
				return m.costMaxOnly(desired, actual, n)
			} else if m == WestonWatkins {
				return m.costSum(desired, actual, n)
			} else {
				panic("invalid MultiHinge")
			}
		})
	})
}

func (m MultiHinge) costMaxOnly(desired, actual anydiff.Res, n int) anydiff.Res {
	// We ignore the desired output via scaling by 0.
	// In order for that to work, every vector component
	// must be non-negative.
	offset := anyvec.AbsMax(actual.Output())
	actual = anydiff.AddScaler(actual, offset)

	cols := desired.Output().Len() / n
	rights := anydiff.SumCols(
		&anydiff.Matrix{
			Data: anydiff.Mul(desired, actual),
			Rows: n,
			Cols: cols,
		},
	)

	maskedOut := desired.Output().Copy()
	anyvec.Complement(maskedOut)
	maskedOut.Mul(actual.Output())
	maxMap := anyvec.MapMax(maskedOut, cols)

	wrongs := anydiff.Map(maxMap, actual)
	diffs := anydiff.Sub(wrongs, rights)
	one := diffs.Output().Creator().MakeNumeric(1)
	return anydiff.ClipPos(anydiff.AddScaler(diffs, one))
}

func (m MultiHinge) costSum(desired, actual anydiff.Res, n int) anydiff.Res {
	cols := desired.Output().Len() / n
	rights := anydiff.SumCols(
		&anydiff.Matrix{
			Data: anydiff.Mul(desired, actual),
			Rows: n,
			Cols: cols,
		},
	)

	mask := anydiff.Complement(desired)

	repeated := anydiff.ScaleRows(&anydiff.Matrix{
		Data: mask,
		Rows: n,
		Cols: cols,
	}, rights).Data
	differences := anydiff.Sub(actual, repeated)
	margins := anydiff.AddScaler(differences, mask.Output().Creator().MakeNumeric(1))
	subCosts := anydiff.ClipPos(anydiff.Mul(margins, mask))

	return anydiff.SumCols(&anydiff.Matrix{Data: subCosts, Rows: n, Cols: cols})
}
