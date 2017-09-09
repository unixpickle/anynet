// Package anyconv provides various types of layers for
// convolutional neural networks.
package anyconv

import (
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
)

// A Conver implements a specific convolution operation,
// where the filter sizes and input sizes are already
// determined.
type Conver interface {
	anynet.Layer
}

// A ConverMaker constructs new Convers for a given set of
// layer parameters.
//
// Different ConverMakers may use different convolution
// algorithms.
// For example, you might want to implement a ConverMaker
// that produces Convers that use GPU-specific routines.
type ConverMaker func(info Conv) Conver

var converMakerLock sync.RWMutex
var converMaker ConverMaker = MakeDefaultConver

// SetConverMaker sets the function which should be used
// to create new Convers.
// The maker is used when deserializing Conv layers or
// when parsing markup.
func SetConverMaker(f ConverMaker) {
	converMakerLock.Lock()
	converMaker = f
	converMakerLock.Unlock()
}

// CurrentConverMaker returns the current function for
// creating Convers.
func CurrentConverMaker() ConverMaker {
	converMakerLock.RLock()
	defer converMakerLock.RUnlock()
	return converMaker
}

// MakeDefaultConver is the default ConverMaker.
// It returns Convers that use anyvec primitives to
// perform convolution.
func MakeDefaultConver(c Conv) Conver {
	if c.Biases == nil || c.Filters == nil {
		panic("nil parameters")
	}
	return &conver{
		conv: c,
		im2row: &Im2Row{
			WindowWidth:  c.FilterWidth,
			WindowHeight: c.FilterHeight,

			StrideX: c.StrideX,
			StrideY: c.StrideY,

			InputWidth:  c.InputWidth,
			InputHeight: c.InputHeight,
			InputDepth:  c.InputDepth,
		},
	}
}

// MakeParallelConver is similar to MakeDefaultConver,
// except that the resulting Conver will parallelize
// convolutions across the batch.
// This is good for running small convolutions on a CPU
// efficiently.
func MakeParallelConver(c Conv) Conver {
	res := MakeDefaultConver(c).(*conver)
	res.parallel = true
	return res
}

// conver is the default Conver implementation.
//
// It uses the classic "im2col technique", where input
// tensors are converted to matrices.
type conver struct {
	conv   Conv
	im2row *Im2Row

	parallel bool
}

// Apply applies the layer an input tensor.
//
// The layer must have been initialized.
//
// After you apply a Conv, you should not modify
// its fields again.
func (c *conver) Apply(in anydiff.Res, batchSize int) anydiff.Res {
	if c.conv.OutputWidth() == 0 || c.conv.OutputHeight() == 0 {
		return anydiff.NewConst(in.Output().Creator().MakeVector(0))
	}
	imgSize := c.conv.InputWidth * c.conv.InputHeight * c.conv.InputDepth
	if in.Output().Len() != batchSize*imgSize {
		panic("incorrect input size")
	}

	filterMatrix := c.filterMatrix()
	outImgSize := c.conv.OutputWidth() * c.conv.OutputHeight() * c.conv.OutputDepth()

	productResults := make([]anyvec.Vector, batchSize)
	c.mapper()(in.Output(), func(i int, imgMatrix *anyvec.Matrix) {
		prodMat := &anyvec.Matrix{
			Data: in.Output().Creator().MakeVector(outImgSize),
			Rows: c.conv.OutputWidth() * c.conv.OutputHeight(),
			Cols: c.conv.OutputDepth(),
		}
		prodMat.Product(false, true, in.Output().Creator().MakeNumeric(1),
			imgMatrix, filterMatrix, in.Output().Creator().MakeNumeric(0))
		productResults[i] = prodMat.Data
	})

	outData := in.Output().Creator().Concat(productResults...)
	anyvec.AddRepeated(outData, c.conv.Biases.Vector)

	ourVars := anydiff.VarSet{}
	ourVars.Add(c.conv.Filters)
	ourVars.Add(c.conv.Biases)
	merged := anydiff.MergeVarSets(in.Vars(), ourVars)

	return &convRes{
		Conver: c,
		Layer:  &c.conv,
		N:      batchSize,
		In:     in,
		OutVec: outData,
		V:      merged,
	}
}

func (c *conver) filterMatrix() *anyvec.Matrix {
	return &anyvec.Matrix{
		Data: c.conv.Filters.Vector,
		Rows: c.conv.FilterCount,
		Cols: c.conv.FilterWidth * c.conv.FilterHeight * c.conv.InputDepth,
	}
}

func (c *conver) mapper() func(anyvec.Vector, func(int, *anyvec.Matrix)) {
	if c.parallel {
		return c.im2row.MapParallel
	} else {
		return c.im2row.MapAll
	}
}

type convRes struct {
	Conver *conver
	Layer  *Conv
	N      int
	In     anydiff.Res
	OutVec anyvec.Vector
	V      anydiff.VarSet
}

func (c *convRes) Output() anyvec.Vector {
	return c.OutVec
}

func (c *convRes) Vars() anydiff.VarSet {
	return c.V
}

func (c *convRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	doIn := g.Intersects(c.In.Vars())

	outSize := u.Len() / c.N
	inSize := c.In.Output().Len() / c.N

	filterMat := c.Conver.filterMatrix()

	one := u.Creator().MakeNumeric(1)
	zero := u.Creator().MakeNumeric(0)

	if biasGrad, ok := g[c.Layer.Biases]; ok {
		c.propagateBiases(biasGrad, u)
	}

	inputUpstreams := make([]anyvec.Vector, c.N)
	var updateLock sync.Mutex
	c.loopImageMatrix(g, func(i int, imgMat *anyvec.Matrix) {
		uMat := &anyvec.Matrix{
			Data: u.Slice(outSize*i, outSize*(i+1)),
			Rows: c.Layer.OutputWidth() * c.Layer.OutputHeight(),
			Cols: c.Layer.OutputDepth(),
		}
		if filterGrad, ok := g[c.Layer.Filters]; ok {
			fgMat := *filterMat
			fgMat.Data = filterGrad.Creator().MakeVector(filterGrad.Len())
			fgMat.Product(true, false, one, uMat, imgMat, zero)
			updateLock.Lock()
			filterGrad.Add(fgMat.Data)
			updateLock.Unlock()
		}
		if doIn {
			imgMat.Product(false, false, one, uMat, filterMat, zero)
			inUp := u.Creator().MakeVector(inSize)
			c.Conver.im2row.Mapper(u.Creator()).MapTranspose(imgMat.Data, inUp)
			inputUpstreams[i] = inUp
		}
	})

	if doIn {
		totalUp := u.Creator().Concat(inputUpstreams...)
		c.In.Propagate(totalUp, g)
	}
}

func (c *convRes) loopImageMatrix(g anydiff.Grad, f func(i int, m *anyvec.Matrix)) {
	if _, ok := g[c.Layer.Filters]; ok {
		c.Conver.mapper()(c.In.Output(), f)
	} else if c.Conver.parallel {
		c.Conver.im2row.CallParallel(c.In.Output().Creator(), c.N, f)
	} else {
		c.Conver.im2row.CallAll(c.In.Output().Creator(), c.N, f)
	}
}

func (c *convRes) propagateBiases(biasGrad, upstream anyvec.Vector) {
	upMat := &anyvec.Matrix{
		Data: upstream,
		Rows: upstream.Len() / c.Layer.Biases.Vector.Len(),
		Cols: c.Layer.Biases.Vector.Len(),
	}
	oneMat := &anyvec.Matrix{
		Data: upstream.Creator().MakeVector(upMat.Rows),
		Rows: upMat.Rows,
		Cols: 1,
	}
	oneMat.Data.AddScalar(upstream.Creator().MakeNumeric(1))
	resMat := &anyvec.Matrix{
		Data: biasGrad,
		Rows: biasGrad.Len(),
		Cols: 1,
	}
	one := upstream.Creator().MakeNumeric(1)
	resMat.Product(true, false, one, upMat, oneMat, one)
}
