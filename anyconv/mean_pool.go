package anyconv

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/serializer"
)

func init() {
	var m MeanPool
	serializer.RegisterTypedDeserializer(m.SerializerType(), DeserializeMeanPool)
}

// MeanPool is a mean-pooling layer.
//
// It behaves very similarly to MaxPool.
type MeanPool struct {
	SpanX int
	SpanY int

	StrideX int
	StrideY int

	InputWidth  int
	InputHeight int
	InputDepth  int

	im2col anyvec.Mapper
}

// DeserializeMeanPool deserializes a MeanPool.
func DeserializeMeanPool(d []byte) (*MeanPool, error) {
	mp, err := DeserializeMaxPool(d)
	if err != nil {
		return nil, err
	}
	return &MeanPool{
		SpanX:       mp.SpanX,
		SpanY:       mp.SpanY,
		StrideX:     mp.StrideX,
		StrideY:     mp.StrideY,
		InputWidth:  mp.InputWidth,
		InputHeight: mp.InputHeight,
		InputDepth:  mp.InputDepth,
	}, nil
}

// OutputWidth returns the output tensor width.
func (m *MeanPool) OutputWidth() int {
	return m.maxPool().OutputWidth()
}

// OutputHeight returns the output tensor height.
func (m *MeanPool) OutputHeight() int {
	return m.maxPool().OutputHeight()
}

// OutputDepth returns the depth of the output tensor.
func (m *MeanPool) OutputDepth() int {
	return m.InputDepth
}

// Apply applies the pooling layer.
func (m *MeanPool) Apply(in anydiff.Res, batchSize int) anydiff.Res {
	if m.im2col == nil {
		m.initIm2Col(in.Output().Creator())
	}
	imgSize := m.im2col.InSize()
	if in.Output().Len() != batchSize*imgSize {
		panic("incorrect input size")
	}

	outCount := m.OutputHeight() * m.OutputWidth() * m.OutputDepth()
	im2ColTemp := in.Output().Creator().MakeVector(m.im2col.OutSize())

	var sumResults []anyvec.Vector
	for i := 0; i < batchSize; i++ {
		subIn := in.Output().Slice(imgSize*i, imgSize*(i+1))
		m.im2col.Map(subIn, im2ColTemp)
		sum := anyvec.SumCols(im2ColTemp, outCount)
		sumResults = append(sumResults, sum)
	}

	out := in.Output().Creator().Concat(sumResults...)
	scaler := out.Creator().MakeNumeric(1 / float64(m.SpanX*m.SpanY))
	out.Scale(scaler)

	return &meanPoolRes{
		Layer:  m,
		N:      batchSize,
		In:     in,
		Scaler: scaler,
		OutVec: out,
	}
}

// SerializerType returns the unique ID used to serialize
// a MeanPool with the serializer package.
func (m *MeanPool) SerializerType() string {
	return "github.com/unixpickle/anynet/anyconv.MeanPool"
}

// Serialize serializes the layer.
func (m *MeanPool) Serialize() ([]byte, error) {
	return m.maxPool().Serialize()
}

func (m *MeanPool) maxPool() *MaxPool {
	return &MaxPool{
		SpanX:       m.SpanX,
		SpanY:       m.SpanY,
		StrideX:     m.StrideX,
		StrideY:     m.StrideY,
		InputWidth:  m.InputWidth,
		InputHeight: m.InputHeight,
		InputDepth:  m.InputDepth,
	}
}

func (m *MeanPool) initIm2Col(c anyvec.Creator) {
	mp := m.maxPool()
	mp.initIm2Col(c)
	m.im2col = mp.im2col
}

type meanPoolRes struct {
	Layer  *MeanPool
	N      int
	In     anydiff.Res
	Scaler anyvec.Numeric
	OutVec anyvec.Vector
}

func (m *meanPoolRes) Output() anyvec.Vector {
	return m.OutVec
}

func (m *meanPoolRes) Vars() anydiff.VarSet {
	return m.In.Vars()
}

func (m *meanPoolRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	// Scaling u first is more efficient.
	u.Scale(m.Scaler)

	mappedU := u.Creator().MakeVector(m.Layer.im2col.OutSize() * m.N)
	anyvec.AddChunks(mappedU, u)

	m.In.Propagate(batchMapTranspose(m.Layer.im2col, mappedU), g)
}
