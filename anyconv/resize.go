package anyconv

import (
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var r Resize
	serializer.RegisterTypedDeserializer(r.SerializerType(), DeserializeResize)
}

// A Resize layer resizes tensors using bilinear
// interpolation.
//
// The output dimensions must be greater than 1.
type Resize struct {
	Depth int

	InputWidth   int
	InputHeight  int
	OutputWidth  int
	OutputHeight int

	mappingLock     sync.Mutex
	neighborMap     anyvec.Mapper
	neighborWeights anyvec.Vector
}

// DeserializeResize deserializes a Resize.
func DeserializeResize(d []byte) (*Resize, error) {
	var depth, inW, inH, outW, outH serializer.Int
	err := serializer.DeserializeAny(d, &depth, &inW, &inH, &outW, &outH)
	if err != nil {
		return nil, essentials.AddCtx("deserialize Resize", err)
	}
	return &Resize{
		Depth:        int(depth),
		InputWidth:   int(inW),
		InputHeight:  int(inH),
		OutputWidth:  int(outW),
		OutputHeight: int(outH),
	}, nil
}

// Apply applies the layer to an input tensor.
func (r *Resize) Apply(in anydiff.Res, batchSize int) anydiff.Res {
	if batchSize == 0 {
		return anydiff.NewConst(in.Output().Creator().MakeVector(0))
	}
	if r.InputWidth == 0 || r.InputHeight == 0 || r.OutputWidth <= 1 ||
		r.OutputHeight <= 1 || r.Depth == 0 {
		panic("tensor dimension out of range")
	}
	if r.InputWidth*r.InputHeight*r.Depth*batchSize != in.Output().Len() {
		panic("incorrect input size")
	}

	r.mappingLock.Lock()
	if r.neighborMap == nil {
		r.initializeMapping(in.Output().Creator())
	}
	r.mappingLock.Unlock()

	mapped := batchMap(r.neighborMap, in.Output())
	anyvec.ScaleRepeated(mapped, r.neighborWeights)
	out := anyvec.SumCols(mapped, mapped.Len()/4)

	return &resizeRes{
		Layer: r,
		In:    in,
		Out:   out,
		Batch: batchSize,
	}
}

// SerializerType returns the unique ID used to serialize
// a Resize with the serializer package.
func (r *Resize) SerializerType() string {
	return "github.com/unixpickle/anynet/anyconv.Resize"
}

// Serialize serializes the Resize.
func (r *Resize) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		serializer.Int(r.Depth),
		serializer.Int(r.InputWidth),
		serializer.Int(r.InputHeight),
		serializer.Int(r.OutputWidth),
		serializer.Int(r.OutputHeight),
	)
}

func (r *Resize) initializeMapping(c anyvec.Creator) {
	var sources []int
	var amounts []float64
	xScale := float64(r.InputWidth-1) / float64(r.OutputWidth-1)
	yScale := float64(r.InputHeight-1) / float64(r.OutputHeight-1)
	for y := 0; y < r.OutputHeight; y++ {
		sourceY := yScale * float64(y)
		for x := 0; x < r.OutputWidth; x++ {
			sourceX := xScale * float64(x)
			neighbors, a := r.neighbors(sourceX, sourceY)
			for z := 0; z < r.Depth; z++ {
				for _, idx := range neighbors[:] {
					sources = append(sources, idx+z)
				}
				amounts = append(amounts, a[:]...)
			}
		}
	}
	r.neighborMap = c.MakeMapper(r.InputWidth*r.InputHeight*r.Depth, sources)
	r.neighborWeights = c.MakeVectorData(c.MakeNumericList(amounts))
}

func (r *Resize) neighbors(sx, sy float64) ([4]int, [4]float64) {
	if sx > float64(r.InputWidth-1) {
		sx = float64(r.InputWidth - 1)
	}
	if sy > float64(r.InputHeight-1) {
		sy = float64(r.InputHeight - 1)
	}
	x1, x2 := int(sx), int(sx+1)
	y1, y2 := int(sy), int(sy+1)
	if x1 < 0 || y1 < 0 {
		x1 = 0
		y1 = 0
	}
	if x2 >= r.InputWidth || y2 >= r.InputHeight {
		x2 = r.InputWidth - 1
		y2 = r.InputHeight - 1
	}

	x1A := 1 - (sx - float64(x1))
	y1A := 1 - (sy - float64(y1))

	return [4]int{
			r.sourceIndex(x1, y1),
			r.sourceIndex(x2, y1),
			r.sourceIndex(x1, y2),
			r.sourceIndex(x2, y2),
		}, [4]float64{
			x1A * y1A,
			(1 - x1A) * y1A,
			x1A * (1 - y1A),
			(1 - x1A) * (1 - y1A),
		}
}

func (r *Resize) sourceIndex(x, y int) int {
	return r.Depth * (x + r.InputWidth*y)
}

type resizeRes struct {
	Layer *Resize
	In    anydiff.Res
	Out   anyvec.Vector
	Batch int
}

func (r *resizeRes) Output() anyvec.Vector {
	return r.Out
}

func (r *resizeRes) Vars() anydiff.VarSet {
	return r.In.Vars()
}

func (r *resizeRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	repSize := r.Layer.neighborWeights.Len() * r.Batch
	mappedDown := r.Layer.neighborWeights.Creator().MakeVector(repSize)
	anyvec.AddRepeated(mappedDown, r.Layer.neighborWeights)
	anyvec.ScaleChunks(mappedDown, u)
	down := batchMapTranspose(r.Layer.neighborMap, mappedDown)
	r.In.Propagate(down, g)
}
