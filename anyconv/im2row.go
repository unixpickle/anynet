package anyconv

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/unixpickle/anyvec"
)

// Im2Row maps (possibly overlapping) regions in an input
// tensor to rows in a matrix.
// The regions are defined by sliding a window of size
// WindowWidth by WindowHeight along the image with a
// stride of StrideX and StrideY.
//
// Each row corresponds to an (x,y) coordinate in the
// output tensor for a Conv, MaxPool, or MeanPool.
// In particular, the i-th row corresponds to the i-th
// (x,y) coordinate in the output tensor for a Conv,
// MaxPool, or MeanPool.
//
// An Im2Row may cache various information about the
// mapping.
// You should not modify an Im2Row after using it for any
// mapping operation.
type Im2Row struct {
	WindowWidth  int
	WindowHeight int

	StrideX int
	StrideY int

	InputWidth  int
	InputHeight int
	InputDepth  int

	mapperLock sync.Mutex
	mapper     anyvec.Mapper
}

// InputSize returns the total number of components in the
// input tensors.
func (m *Im2Row) InputSize() int {
	return m.InputWidth * m.InputHeight * m.InputDepth
}

// NumX returns the number of horizontal sliding window
// positions.
// This is also the width of the output tensors produced
// by a Conv with the parameters of m.
func (m *Im2Row) NumX() int {
	w := 1 + (m.InputWidth-m.WindowWidth)/m.StrideX
	if w < 0 {
		return 0
	} else {
		return w
	}
}

// NumY returns the number of vertical sliding window
// positions.
// This is also the height of the output tensors produced
// by a Conv with the parameters of m.
func (m *Im2Row) NumY() int {
	h := 1 + (m.InputHeight-m.WindowHeight)/m.StrideY
	if h < 0 {
		return 0
	} else {
		return h
	}
}

// MakeOut allocates a row matrix for the output of Map.
func (m *Im2Row) MakeOut(c anyvec.Creator) *anyvec.Matrix {
	rows := m.NumX() * m.NumY()
	cols := m.WindowWidth * m.WindowHeight * m.InputDepth
	return &anyvec.Matrix{Data: c.MakeVector(rows * cols), Rows: rows, Cols: cols}
}

// MapAll maps one or more input tensor to a row matrix.
//
// The f func is called for every input tensor in order.
// It may modify the contents of m.Data, but it should
// not keep a reference to the matrix argument after it
// returns, since the matrix's data vector may be reused
// for every input tensor.
//
// The length of in must be divisible by the input tensor
// described by m.InputWidth and others.
func (m *Im2Row) MapAll(in anyvec.Vector, f func(idx int, m *anyvec.Matrix)) {
	m.mapImpl(in, f, false)
}

// MapParallel is like MapAll, except that f may be called
// multiple times concurrently and the calls are not
// necessarily in order.
func (m *Im2Row) MapParallel(in anyvec.Vector, f func(idx int, m *anyvec.Matrix)) {
	m.mapImpl(in, f, true)
}

func (m *Im2Row) mapImpl(in anyvec.Vector, f func(idx int, m *anyvec.Matrix),
	parallel bool) {
	inSize := m.InputSize()
	if in.Len()%inSize != 0 {
		panic(fmt.Sprintf("input length %d not divisible by %d", in.Len(), inSize))
	}

	mapper := m.Mapper(in.Creator())
	mapAndCall := func(i int, m *anyvec.Matrix) {
		subIn := in.Slice(inSize*i, inSize*(i+1))
		mapper.Map(subIn, m.Data)
		f(i, m)
	}

	n := in.Len() / inSize
	if parallel {
		m.CallParallel(in.Creator(), n, mapAndCall)
	} else {
		m.CallAll(in.Creator(), n, mapAndCall)
	}
}

// CallAll is like MapAll, except it doesn't perform the
// mapping itself.
// Rather, it allocates a matrix, but does not write
// anything in particular to the matrix.
// The matrix may contain arbitrary junk when it is passed
// to f.
func (m *Im2Row) CallAll(c anyvec.Creator, n int, f func(int, *anyvec.Matrix)) {
	imageMat := m.MakeOut(c)
	for i := 0; i < n; i++ {
		f(i, imageMat)
	}
}

// CallParallel is like MapParallel, except it doesn't
// perform the mapping itself.
//
// See CallAll for details.
func (m *Im2Row) CallParallel(c anyvec.Creator, n int, f func(int, *anyvec.Matrix)) {
	jobs := make(chan int, n)
	for i := 0; i < n; i++ {
		jobs <- i
	}
	close(jobs)

	var wg sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			imageMat := m.MakeOut(c)
			for i := range jobs {
				f(i, imageMat)
			}
		}()
	}

	wg.Wait()
}

// Mapper returns a mapper for the mapping.
func (m *Im2Row) Mapper(c anyvec.Creator) anyvec.Mapper {
	m.mapperLock.Lock()
	defer m.mapperLock.Unlock()
	if m.mapper != nil && m.mapper.Creator() == c {
		return m.mapper
	}

	var mapping []int

	for y := 0; y+m.WindowHeight <= m.InputHeight; y += m.StrideY {
		for x := 0; x+m.WindowWidth <= m.InputWidth; x += m.StrideX {
			for subY := 0; subY < m.WindowHeight; subY++ {
				subYIdx := (y + subY) * m.InputWidth * m.InputDepth
				for subX := 0; subX < m.WindowWidth; subX++ {
					subXIdx := subYIdx + (subX+x)*m.InputDepth
					for subZ := 0; subZ < m.InputDepth; subZ++ {
						mapping = append(mapping, subXIdx+subZ)
					}
				}
			}
		}
	}

	inSize := m.InputWidth * m.InputHeight * m.InputDepth
	m.mapper = c.MakeMapper(inSize, mapping)

	return m.mapper
}
