package anyconv

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var p Padding
	serializer.RegisterTypedDeserializer(p.SerializerType(), DeserializePadding)
}

// A Padding layer adds zeros to the border of input
// tensors.
type Padding struct {
	InputWidth  int
	InputHeight int
	InputDepth  int

	PaddingTop    int
	PaddingRight  int
	PaddingBottom int
	PaddingLeft   int

	mapper anyvec.Mapper
}

// DeserializePadding deserializes a Padding.
func DeserializePadding(d []byte) (*Padding, error) {
	var inW, inH, inD, pT, pR, pB, pL serializer.Int
	err := serializer.DeserializeAny(d, &inW, &inH, &inD, &pT, &pR, &pB, &pL)
	if err != nil {
		return nil, essentials.AddCtx("deserialize Padding", err)
	}
	return &Padding{
		InputWidth:  int(inW),
		InputHeight: int(inH),
		InputDepth:  int(inD),

		PaddingTop:    int(pT),
		PaddingRight:  int(pR),
		PaddingBottom: int(pB),
		PaddingLeft:   int(pL),
	}, nil
}

// Apply applies the layer.
//
// This is not thread-safe.
func (p *Padding) Apply(in anydiff.Res, batch int) anydiff.Res {
	if p.mapper == nil {
		p.initMapper(in.Output().Creator())
	}
	if in.Output().Len() != batch*p.mapper.OutSize() {
		panic("incorrect input size")
	}
	return &paddingRes{
		In:     in,
		Mapper: p.mapper,
		OutVec: batchMapTranspose(p.mapper, in.Output()),
	}
}

// SerializerType returns the unique ID used to serialize
// a Padding with the serializer package.
func (p *Padding) SerializerType() string {
	return "github.com/unixpickle/anynet/anyconv.Padding"
}

// Serialize serializes a Padding.
func (p *Padding) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		serializer.Int(p.InputWidth),
		serializer.Int(p.InputHeight),
		serializer.Int(p.InputDepth),
		serializer.Int(p.PaddingTop),
		serializer.Int(p.PaddingRight),
		serializer.Int(p.PaddingBottom),
		serializer.Int(p.PaddingLeft),
	)
}

func (p *Padding) initMapper(c anyvec.Creator) {
	newWidth := p.InputWidth + p.PaddingLeft + p.PaddingRight
	outSize := newWidth * (p.InputHeight + p.PaddingTop + p.PaddingBottom) * p.InputDepth
	table := make([]int, 0, p.InputWidth*p.InputHeight*p.InputDepth)

	for y := 0; y < p.InputHeight; y++ {
		yOffset := (y + p.PaddingTop) * newWidth * p.InputDepth
		for x := 0; x < p.InputWidth; x++ {
			xOffset := yOffset + (x+p.PaddingLeft)*p.InputDepth
			for z := 0; z < p.InputDepth; z++ {
				table = append(table, xOffset+z)
			}
		}
	}

	p.mapper = c.MakeMapper(outSize, table)
}

type paddingRes struct {
	In     anydiff.Res
	Mapper anyvec.Mapper
	OutVec anyvec.Vector
}

func (p *paddingRes) Output() anyvec.Vector {
	return p.OutVec
}

func (p *paddingRes) Vars() anydiff.VarSet {
	return p.In.Vars()
}

func (p *paddingRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	p.In.Propagate(batchMap(p.Mapper, u), g)
}
