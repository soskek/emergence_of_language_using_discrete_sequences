for imp in 0 1e-1 1e-2 1e-3
do
    for ort in 0 1e-1 1e-2 1e-3
    do
        for w in 3 10
        do
            for unit in 32 64 128 256
            do
                for vocab in 2 4
                do
                    echo logs/log.ort${ort}imp${imp}w${w}t1b512iu256u${unit}d02v${vocab}.log &
                    python -u scripts/train_mnist.py --gpu=0 --epoch 200 --batchsize 512 --out data0306_v2/test_ort${ort}imp${imp}w${w}t1b512iu256u${unit}d02v${vocab} -ort ${ort} -imp ${imp} --image-unit 256 --unit ${unit} --word ${w} --turn 1 --dropout 0.2 --vocab ${vocab} > logs0306_v2/log.ort${ort}imp${imp}w${w}t1b512iu256u${unit}d02v${vocab}.log &
                done
            done
        done
        python -u scripts/train_mnist.py --gpu=0 --epoch 190 --batchsize 512 --out gomidir -ort 1e-3 -imp 1e-3 --image-unit 256 --unit 32 --word 5 --turn 1 --dropout 0.2 -v 32 > log.gomi

        for w in 15
        do
            for unit in 64 128
            do
                for vocab in 2 4
                do
                    echo logs/log.ort${ort}imp${imp}w${w}t1b512iu256u${unit}d02v${vocab}.log &
                    python -u scripts/train_mnist.py --gpu=0 --epoch 200 --batchsize 512 --out data0306_v2/test_ort${ort}imp${imp}w${w}t1b512iu256u${unit}d02v${vocab} -ort ${ort} -imp ${imp} --image-unit 256 --unit ${unit} --word ${w} --turn 1 --dropout 0.2 --vocab ${vocab} > logs0306_v2/log.ort${ort}imp${imp}w${w}t1b512iu256u${unit}d02v${vocab}.log &
                done
            done
        done
        python -u scripts/train_mnist.py --gpu=0 --epoch 190 --batchsize 512 --out gomidir -ort 1e-3 -imp 1e-3 --image-unit 256 --unit 32 --word 5 --turn 1 --dropout 0.2 -v 8 > log.gomi

        for w in 15
        do
            for unit in 32 256
            do
                for vocab in 2 4
                do
                    echo logs/log.ort${ort}imp${imp}w${w}t1b512iu256u${unit}d02v${vocab}.log &
                    python -u scripts/train_mnist.py --gpu=0 --epoch 200 --batchsize 512 --out data0306_v2/test_ort${ort}imp${imp}w${w}t1b512iu256u${unit}d02v${vocab} -ort ${ort} -imp ${imp} --image-unit 256 --unit ${unit} --word ${w} --turn 1 --dropout 0.2 --vocab ${vocab} > logs0306_v2/log.ort${ort}imp${imp}w${w}t1b512iu256u${unit}d02v${vocab}.log &
                done
            done
        done
        python -u scripts/train_mnist.py --gpu=0 --epoch 190 --batchsize 512 --out gomidir -ort 1e-3 -imp 1e-3 --image-unit 256 --unit 32 --word 5 --turn 1 --dropout 0.2 -v 8 > log.gomi
    done
done
