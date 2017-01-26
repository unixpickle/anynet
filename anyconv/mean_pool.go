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
// If a span does not divide the corresponding input
// dimension, the input is zero padded.
//
// All input and output tensors are row-major depth-minor.
type MeanPool struct {
	SpanX int
	SpanY int

	InputWidth  int
	InputHeight int
	InputDepth  int

	mapper anyvec.Mapper
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
		InputWidth:  mp.InputWidth,
		InputHeight: mp.InputHeight,
		InputDepth:  mp.InputDepth,
	}, nil
}

// OutputWidth returns the output tensor width.
func (m *MeanPool) OutputWidth() int {
	return (m.InputWidth + m.SpanX - 1) / m.SpanX
}

// OutputHeight returns the output tensor height.
func (m *MeanPool) OutputHeight() int {
	return (m.InputHeight + m.SpanY - 1) / m.SpanY
}

// OutputDepth returns the depth of the output tensor.
func (m *MeanPool) OutputDepth() int {
	return m.InputDepth
}

// Apply applies the pooling layer.
func (m *MeanPool) Apply(in anydiff.Res, batchSize int) anydiff.Res {
	if m.mapper == nil {
		m.initMapper(in.Output().Creator())
	}
	if in.Output().Len() != batchSize*m.mapper.OutSize() {
		panic("incorrect input size")
	}
	out := batchMapTranspose(m.mapper, in.Output())
	scaler := out.Creator().MakeNumeric(1 / float64(m.SpanX*m.SpanY))
	out.Scale(scaler)
	return &meanPoolRes{
		In:     in,
		Mapper: m.mapper,
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
		InputWidth:  m.InputWidth,
		InputHeight: m.InputHeight,
		InputDepth:  m.InputDepth,
	}
}

func (m *MeanPool) initMapper(c anyvec.Creator) {
	table := make([]int, 0, m.InputWidth*m.InputHeight*m.InputDepth)
	outWidth := m.OutputWidth()
	for y := 0; y < m.InputHeight; y++ {
		yIdx := (y / m.SpanY) * outWidth * m.InputDepth
		for x := 0; x < m.InputWidth; x++ {
			xIdx := yIdx + (x/m.SpanX)*m.InputDepth
			for z := 0; z < m.InputDepth; z++ {
				table = append(table, xIdx+z)
			}
		}
	}
	m.mapper = c.MakeMapper(outWidth*m.OutputHeight()*m.OutputDepth(), table)
}

type meanPoolRes struct {
	In     anydiff.Res
	Mapper anyvec.Mapper
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

	m.In.Propagate(batchMap(m.Mapper, u), g)
}
