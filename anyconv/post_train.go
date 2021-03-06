package anyconv

import (
	"errors"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

// A PostTrainer uses a list of samples to replace
// BatchNorm layers with hard-wired affine transforms.
// It automatically looks for BatchNorm layers inside
// Residual blocks.
type PostTrainer struct {
	Samples anysgd.SampleList

	// Fetcher must return *anyff.Batch instances.
	Fetcher anysgd.Fetcher

	// BatchSize specifies how many samples to feed to the
	// network at once.
	BatchSize int

	Net anynet.Net

	// StatusFunc, if non-nil, is called for every BatchNorm
	// layer right before the layer is replaced.
	StatusFunc func(bn *BatchNorm)
}

// Run performs layer replacement.
//
// It returns with an error if the Fetcher failed.
// However, even in the event of an error, some layers may
// already be replaced with affine transforms.
func (p *PostTrainer) Run() error {
	return affinizeNet(p.Net, func(bn *BatchNorm, subNet anynet.Net) (*anynet.Affine, error) {
		mean, stddev, err := momentsFromOutputs(bn, p.evaluateBatch(subNet))
		if err != nil {
			return nil, essentials.AddCtx("post train", err)
		}
		scaler := bn.Scalers.Vector.Copy()
		scaler.Div(stddev)
		bias := bn.Biases.Vector.Copy()
		mean.Mul(scaler)
		bias.Sub(mean)
		if p.StatusFunc != nil {
			p.StatusFunc(bn)
		}
		return &anynet.Affine{
			Scalers: anydiff.NewVar(scaler),
			Biases:  anydiff.NewVar(bias),
		}, nil
	})
}

func (p *PostTrainer) evaluateBatch(subNet anynet.Net) <-chan *postTrainerOutput {
	resChan := make(chan *postTrainerOutput, 1)
	batchChan := make(chan anysgd.Batch, 1)
	batchSizesChan := make(chan int, 1)
	batchErrChan := make(chan error, 1)
	go func() {
		defer close(batchChan)
		defer close(batchSizesChan)
		defer close(batchErrChan)
		for i := 0; i < p.Samples.Len(); i += p.BatchSize {
			bs := p.Samples.Len() - i
			if bs > p.BatchSize {
				bs = p.BatchSize
			}
			slice := p.Samples.Slice(i, i+bs)
			if batch, err := p.Fetcher.Fetch(slice); err == nil {
				batchChan <- batch
				batchSizesChan <- bs
			} else {
				batchErrChan <- err
				return
			}
		}
	}()
	go func() {
		defer close(resChan)
		for batch := range batchChan {
			bs := <-batchSizesChan
			inVec := batch.(*anyff.Batch).Inputs
			outVec := subNet.Apply(inVec, bs).Output().Copy()
			resChan <- &postTrainerOutput{Vec: outVec}
		}
		if err := <-batchErrChan; err != nil {
			resChan <- &postTrainerOutput{Err: err}
		}
	}()
	return resChan
}

func affinizeNet(n anynet.Net, f func(*BatchNorm, anynet.Net) (*anynet.Affine, error)) error {
	for i, layer := range n {
		if bn, ok := layer.(*BatchNorm); ok {
			if a, err := f(bn, n[:i]); err != nil {
				return err
			} else {
				n[i] = a
			}
			continue
		}

		r, ok := layer.(*Residual)
		if !ok {
			continue
		}

		for _, part := range []*anynet.Layer{&r.Layer, &r.Projection} {
			if *part == nil {
				continue
			}
			switch layer := (*part).(type) {
			case anynet.Net:
				subF := func(bn *BatchNorm, subNet anynet.Net) (*anynet.Affine, error) {
					return f(bn, append(append(anynet.Net{}, n[:i]...), subNet...))
				}
				err := affinizeNet(layer, subF)
				if err != nil {
					return err
				}
			case *BatchNorm:
				if a, err := f(layer, n[:i]); err != nil {
					return err
				} else {
					*part = a
				}
			}
		}
	}
	return nil
}

type postTrainerOutput struct {
	Err error
	Vec anyvec.Vector
}

func momentsFromOutputs(b *BatchNorm, c <-chan *postTrainerOutput) (mean,
	stddev anyvec.Vector, err error) {
	var sum, sqSum anyvec.Vector
	var count int
	for item := range c {
		if item.Err != nil {
			return nil, nil, item.Err
		}

		count += item.Vec.Len() / b.InputCount
		thisSum := anyvec.SumRows(item.Vec, b.InputCount)
		item.Vec.Mul(item.Vec.Copy())
		thisSqSum := anyvec.SumRows(item.Vec, b.InputCount)

		if sum == nil {
			sum = thisSum
			sqSum = thisSqSum
		} else {
			sum.Add(thisSum)
			sqSum.Add(thisSqSum)
		}
	}
	if sum == nil {
		return nil, nil, errors.New("no samples to average")
	}
	normalizer := sum.Creator().MakeNumeric(1 / float64(count))
	sum.Scale(normalizer)
	sqSum.Scale(normalizer)

	sumSq := sum.Copy()
	sumSq.Mul(sum)

	sqSum.Sub(sumSq)
	sqSum.AddScalar(sqSum.Creator().MakeNumeric(b.stabilizer()))
	anyvec.Pow(sqSum, sqSum.Creator().MakeNumeric(0.5))

	return sum, sqSum, nil
}
