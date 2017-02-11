// Package anyconv provides various types of layers for
// convolutional neural networks.
package anyconv

import (
	"errors"
	"math"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var c Conv
	serializer.RegisterTypedDeserializer(c.SerializerType(), DeserializeConv)
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

	im2col anyvec.Mapper
}

// DeserializeConv deserialize a Conv.
func DeserializeConv(d []byte) (*Conv, error) {
	var inW, inH, inD, fW, fH, sX, sY serializer.Int
	var f, b *anyvecsave.S
	err := serializer.DeserializeAny(d, &inW, &inH, &inD, &fW, &fH, &sX, &sY, &f, &b)
	if err != nil {
		return nil, essentials.AddCtx("deserialize Conv", err)
	}
	return &Conv{
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
	}, nil
}

// InitRand initializes the layer and randomizes the
// filters.
func (c *Conv) InitRand(cr anyvec.Creator) {
	c.InitZero(cr)

	normalizer := 1 / math.Sqrt(float64(c.FilterWidth*c.FilterHeight*c.InputDepth))
	anyvec.Rand(c.Filters.Vector, anyvec.Normal, nil)
	c.Filters.Vector.Scale(cr.MakeNumeric(normalizer))
}

// InitZero initializes the layer to zero.
func (c *Conv) InitZero(cr anyvec.Creator) {
	filterSize := c.FilterWidth * c.FilterHeight * c.InputDepth
	c.Filters = anydiff.NewVar(cr.MakeVector(filterSize * c.FilterCount))
	c.Biases = anydiff.NewVar(cr.MakeVector(c.FilterCount))
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

// Apply applies the layer an input tensor.
//
// The layer must have been initialized.
//
// This is not thread-safe.
func (c *Conv) Apply(in anydiff.Res, batchSize int) anydiff.Res {
	if c.Filters == nil || c.Biases == nil {
		panic("cannot apply uninitialized Conv")
	}
	if c.im2col == nil {
		c.initIm2Col(c.Filters.Vector.Creator())
	}
	if c.OutputWidth() == 0 || c.OutputHeight() == 0 {
		return anydiff.NewConst(in.Output().Creator().MakeVector(0))
	}
	imgSize := c.InputWidth * c.InputHeight * c.InputDepth
	if in.Output().Len() != batchSize*imgSize {
		panic("incorrect input size")
	}

	imgMatrix := c.im2ColMat()
	filterMatrix := c.filterMatrix()
	outImgSize := c.OutputWidth() * c.OutputHeight() * c.OutputDepth()

	var productResults []anyvec.Vector
	for i := 0; i < batchSize; i++ {
		subIn := in.Output().Slice(imgSize*i, imgSize*(i+1))
		c.im2col.Map(subIn, imgMatrix.Data)
		prodMat := &anyvec.Matrix{
			Data: in.Output().Creator().MakeVector(outImgSize),
			Rows: c.OutputWidth() * c.OutputHeight(),
			Cols: c.OutputDepth(),
		}
		prodMat.Product(false, true, in.Output().Creator().MakeNumeric(1),
			imgMatrix, filterMatrix, in.Output().Creator().MakeNumeric(0))
		productResults = append(productResults, prodMat.Data)
	}

	outData := in.Output().Creator().Concat(productResults...)
	anyvec.AddRepeated(outData, c.Biases.Vector)

	ourVars := anydiff.VarSet{}
	ourVars.Add(c.Filters)
	ourVars.Add(c.Biases)
	merged := anydiff.MergeVarSets(in.Vars(), ourVars)

	return &convRes{
		Layer:  c,
		N:      batchSize,
		In:     in,
		OutVec: outData,
		V:      merged,
	}
}

// Parameters returns the layer's parameters.
// The filters come before the biases in the resulting
// slice.
//
// If the layer is uninitialized, the result is nil
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

func (c *Conv) initIm2Col(cr anyvec.Creator) {
	var mapping []int

	for y := 0; y+c.FilterHeight <= c.InputHeight; y += c.StrideY {
		for x := 0; x+c.FilterWidth <= c.InputWidth; x += c.StrideX {
			for subY := 0; subY < c.FilterHeight; subY++ {
				subYIdx := (y + subY) * c.InputWidth * c.InputDepth
				for subX := 0; subX < c.FilterWidth; subX++ {
					subXIdx := subYIdx + (subX+x)*c.InputDepth
					for subZ := 0; subZ < c.InputDepth; subZ++ {
						mapping = append(mapping, subXIdx+subZ)
					}
				}
			}
		}
	}

	inSize := c.InputWidth * c.InputHeight * c.InputDepth
	c.im2col = cr.MakeMapper(inSize, mapping)
}

func (c *Conv) filterMatrix() *anyvec.Matrix {
	return &anyvec.Matrix{
		Data: c.Filters.Vector,
		Rows: c.FilterCount,
		Cols: c.FilterWidth * c.FilterHeight * c.InputDepth,
	}
}

func (c *Conv) im2ColMat() *anyvec.Matrix {
	return &anyvec.Matrix{
		Data: c.Filters.Vector.Creator().MakeVector(c.im2col.OutSize()),
		Rows: c.im2col.OutSize() / (c.FilterWidth * c.FilterHeight * c.InputDepth),
		Cols: c.FilterWidth * c.FilterHeight * c.InputDepth,
	}
}

type convRes struct {
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

	imgMat := c.Layer.im2ColMat()
	filterMat := c.Layer.filterMatrix()

	one := u.Creator().MakeNumeric(1)
	zero := u.Creator().MakeNumeric(0)

	if biasGrad, ok := g[c.Layer.Biases]; ok {
		c.propagateBiases(biasGrad, u)
	}

	var inputUpstreams []anyvec.Vector
	for i := 0; i < c.N; i++ {
		uMat := &anyvec.Matrix{
			Data: u.Slice(outSize*i, outSize*(i+1)),
			Rows: c.Layer.OutputWidth() * c.Layer.OutputHeight(),
			Cols: c.Layer.OutputDepth(),
		}
		if filterGrad, ok := g[c.Layer.Filters]; ok {
			subIn := c.In.Output().Slice(inSize*i, inSize*(i+1))
			c.Layer.im2col.Map(subIn, imgMat.Data)
			fgMat := *filterMat
			fgMat.Data = filterGrad
			fgMat.Product(true, false, one, uMat, imgMat, one)
		}
		if doIn {
			imgMat.Product(false, false, one, uMat, filterMat, zero)
			inUp := u.Creator().MakeVector(inSize)
			c.Layer.im2col.MapTranspose(imgMat.Data, inUp)
			inputUpstreams = append(inputUpstreams, inUp)
		}
	}

	if doIn {
		totalUp := u.Creator().Concat(inputUpstreams...)
		c.In.Propagate(totalUp, g)
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
