// Package anyconv provides various types of layers for
// convolutional neural networks.
package anyconv

import (
	"errors"
	"math"
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var c Conv
	serializer.RegisterTypedDeserializer(c.SerializerType(), DeserializeConv)
}

// A Conver implements a specific convolution operation,
// where the filter sizes and input sizes are already
// determined.
type Conver interface {
	anynet.Layer
}

// A ConverMaker constructs new Convers for a given set of
// layer parameters.
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

// Conv is a convolutional layer.
//
// All input and output tensors are row-major depth-minor.
type Conv struct {
	FilterCount  int
	FilterWidth  int
	FilterHeight int

	StrideX int
	StrideY int

	InputWidth  int
	InputHeight int
	InputDepth  int

	Filters *anydiff.Var
	Biases  *anydiff.Var

	Conver Conver
}

// DeserializeConv deserialize a Conv.
//
// The Conver is automatically set.
func DeserializeConv(d []byte) (*Conv, error) {
	var inW, inH, inD, fW, fH, sX, sY serializer.Int
	var f, b *anyvecsave.S
	err := serializer.DeserializeAny(d, &inW, &inH, &inD, &fW, &fH, &sX, &sY, &f, &b)
	if err != nil {
		return nil, essentials.AddCtx("deserialize Conv", err)
	}
	res := Conv{
		FilterCount:  f.Vector.Len() / int(fW*fH*inD),
		FilterWidth:  int(fW),
		FilterHeight: int(fH),
		StrideX:      int(sX),
		StrideY:      int(sY),

		InputWidth:  int(inW),
		InputHeight: int(inH),
		InputDepth:  int(inD),

		Filters: anydiff.NewVar(f.Vector),
		Biases:  anydiff.NewVar(b.Vector),
	}
	res.Conver = CurrentConverMaker()(res)
	return &res, nil
}

// InitRand the biases an filters in a randomized fashion
// and sets the Conver.
func (c *Conv) InitRand(cr anyvec.Creator) {
	c.InitZero(cr)

	normalizer := 1 / math.Sqrt(float64(c.FilterWidth*c.FilterHeight*c.InputDepth))
	anyvec.Rand(c.Filters.Vector, anyvec.Normal, nil)
	c.Filters.Vector.Scale(cr.MakeNumeric(normalizer))
}

// InitZero initializes the layer to zero and sets the
// Conver.
func (c *Conv) InitZero(cr anyvec.Creator) {
	filterSize := c.FilterWidth * c.FilterHeight * c.InputDepth
	c.Filters = anydiff.NewVar(cr.MakeVector(filterSize * c.FilterCount))
	c.Biases = anydiff.NewVar(cr.MakeVector(c.FilterCount))
	c.Conver = CurrentConverMaker()(*c)
}

// OutputWidth returns the width of the output tensor.
func (c *Conv) OutputWidth() int {
	w := 1 + (c.InputWidth-c.FilterWidth)/c.StrideX
	if w < 0 {
		return 0
	} else {
		return w
	}
}

// OutputHeight returns the height of the output tensor.
func (c *Conv) OutputHeight() int {
	h := 1 + (c.InputHeight-c.FilterHeight)/c.StrideY
	if h < 0 {
		return 0
	} else {
		return h
	}
}

// OutputDepth returns the depth of the output tensor.
func (c *Conv) OutputDepth() int {
	return c.FilterCount
}

// Apply applies the layer to an input tensor using the
// Conver.
func (c *Conv) Apply(in anydiff.Res, batchSize int) anydiff.Res {
	return c.Conver.Apply(in, batchSize)
}

// Parameters returns the layer's parameters.
// The filters come before the biases in the resulting
// slice.
//
// If the layer is uninitialized, the result is nil.
func (c *Conv) Parameters() []*anydiff.Var {
	if c.Filters == nil || c.Biases == nil {
		return nil
	}
	return []*anydiff.Var{c.Filters, c.Biases}
}

// SerializerType returns the unique ID used to serialize
// a Conv with the serializer package.
func (c *Conv) SerializerType() string {
	return "github.com/unixpickle/anynet/anyconv.Conv"
}

// Serialize serializes the layer.
//
// If the layer was not yet initialized, this fails.
func (c *Conv) Serialize() ([]byte, error) {
	if c.Filters == nil || c.Biases == nil {
		return nil, errors.New("cannot serialize uninitialized Conv")
	}
	return serializer.SerializeAny(
		serializer.Int(c.InputWidth),
		serializer.Int(c.InputHeight),
		serializer.Int(c.InputDepth),
		serializer.Int(c.FilterWidth),
		serializer.Int(c.FilterHeight),
		serializer.Int(c.StrideX),
		serializer.Int(c.StrideY),
		&anyvecsave.S{Vector: c.Filters.Vector},
		&anyvecsave.S{Vector: c.Biases.Vector},
	)
}

type conver struct {
	conv   Conv
	im2row *Im2Row
}

// Apply applies the layer an input tensor.
//
// The layer must have been initialized.
//
// This is not thread-safe.
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

	var productResults []anyvec.Vector
	c.im2row.MapAll(in.Output(), func(_ int, imgMatrix *anyvec.Matrix) {
		prodMat := &anyvec.Matrix{
			Data: in.Output().Creator().MakeVector(outImgSize),
			Rows: c.conv.OutputWidth() * c.conv.OutputHeight(),
			Cols: c.conv.OutputDepth(),
		}
		prodMat.Product(false, true, in.Output().Creator().MakeNumeric(1),
			imgMatrix, filterMatrix, in.Output().Creator().MakeNumeric(0))
		productResults = append(productResults, prodMat.Data)
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

	var inputUpstreams []anyvec.Vector
	c.loopImageMatrix(g, func(i int, imgMat *anyvec.Matrix) {
		uMat := &anyvec.Matrix{
			Data: u.Slice(outSize*i, outSize*(i+1)),
			Rows: c.Layer.OutputWidth() * c.Layer.OutputHeight(),
			Cols: c.Layer.OutputDepth(),
		}
		if filterGrad, ok := g[c.Layer.Filters]; ok {
			fgMat := *filterMat
			fgMat.Data = filterGrad
			fgMat.Product(true, false, one, uMat, imgMat, one)
		}
		if doIn {
			imgMat.Product(false, false, one, uMat, filterMat, zero)
			inUp := u.Creator().MakeVector(inSize)
			c.Conver.im2row.Mapper(u.Creator()).MapTranspose(imgMat.Data, inUp)
			inputUpstreams = append(inputUpstreams, inUp)
		}
	})

	if doIn {
		totalUp := u.Creator().Concat(inputUpstreams...)
		c.In.Propagate(totalUp, g)
	}
}

func (c *convRes) loopImageMatrix(g anydiff.Grad, f func(i int, m *anyvec.Matrix)) {
	if _, ok := g[c.Layer.Filters]; ok {
		c.Conver.im2row.MapAll(c.In.Output(), f)
	} else {
		imgMat := c.Conver.im2row.MakeOut(c.In.Output().Creator())
		for i := 0; i < c.N; i++ {
			f(i, imgMat)
		}
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
	oneMat.Data.AddScaler(upstream.Creator().MakeNumeric(1))
	resMat := &anyvec.Matrix{
		Data: biasGrad,
		Rows: biasGrad.Len(),
		Cols: 1,
	}
	one := upstream.Creator().MakeNumeric(1)
	resMat.Product(true, false, one, upMat, oneMat, one)
}
