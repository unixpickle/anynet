package anynet

import "github.com/unixpickle/anydiff"

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
type SigmoidCE struct{}

// Cost is mathematically equivalent to applying the
// sigmoid to each component of actual, then finding the
// cross-entropy loss.
func (s SigmoidCE) Cost(desired, actual anydiff.Res, n int) anydiff.Res {
	minusOne := actual.Output().Creator().MakeNumeric(-1)
	logRegular := anydiff.LogSigmoid(actual)
	logComplement := anydiff.LogSigmoid(anydiff.Scale(actual, minusOne))
	regularProd := anydiff.Mul(desired, logRegular)
	compProd := anydiff.Mul(anydiff.Complement(desired), logComplement)
	return anydiff.Add(anydiff.Sum(regularProd), anydiff.Sum(compProd))
}
