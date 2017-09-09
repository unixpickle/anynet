// Package anyconv provides various types of layers for
// convolutional neural networks.
package anyconv

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

func TestParallelConver(t *testing.T) {
	c := anyvec64.CurrentCreator()
	conv := Conv{
		FilterCount:  13,
		FilterWidth:  4,
		FilterHeight: 3,

		StrideX: 2,
		StrideY: 3,

		InputWidth:  30,
		InputHeight: 20,
		InputDepth:  7,
	}
	conv.InitRand(c)
	testConverEquiv(t, MakeDefaultConver(conv).(*conver),
		MakeParallelConver(conv).(*conver))
}

func testConverEquiv(t *testing.T, c1, c2 *conver) {
	c := c1.conv.Filters.Vector.Creator()
	inSize := c1.im2row.InputWidth * c1.im2row.InputHeight * c1.im2row.InputDepth
	outSize := c1.conv.OutputWidth() * c1.conv.OutputHeight() * c1.conv.OutputDepth()

	batchSize := 32
	inBatch := c.MakeVector(inSize * batchSize)
	anyvec.Rand(inBatch, anyvec.Normal, nil)
	inVar := anydiff.NewVar(inBatch)

	out1 := c1.Apply(inVar, batchSize)
	out2 := c2.Apply(inVar, batchSize)
	if !vecsClose(out1.Output(), out2.Output()) {
		t.Error("mismatching output values")
	}

	upstream := c.MakeVector(outSize * batchSize)
	anyvec.Rand(upstream, anyvec.Normal, nil)

	vars := []*anydiff.Var{inVar, c1.conv.Filters, c1.conv.Biases}
	grad1 := anydiff.NewGrad(vars...)
	out1.Propagate(upstream.Copy(), grad1)
	grad2 := anydiff.NewGrad(vars...)
	out2.Propagate(upstream.Copy(), grad2)

	for i, variable := range vars {
		g1 := grad1[variable]
		g2 := grad2[variable]
		if !vecsClose(g1, g2) {
			t.Errorf("gradient for variable %d differs", i)
		}
	}
}

func vecsClose(v1, v2 anyvec.Vector) bool {
	c := v1.Creator()
	diff := v1.Copy()
	diff.Sub(v2)
	maxDiff := anyvec.AbsMax(diff)
	thresh := c.MakeNumeric(1e-3)
	return c.NumOps().Less(maxDiff, thresh)
}
