MACROTRAJ="../../test/assets/8state_macrotraj"
MICROTRAJ="../../test/assets/8state_microtraj"
OUTPUT="${MACROTRAJ##*/}"

## WAITING TIME DISTRIBUTION
WTS_ARGS="--lagtimes 1 5 10 --start 1 --final 4 --nsteps 10000000 --frames-per-unit 1 --unit frames" 
python -m msmhelper waiting-times $WTS_ARGS \
    --filename $MACROTRAJ -o ${OUTPUT}.wts.jpg
python -m msmhelper waiting-times $WTS_ARGS \
    --filename $MACROTRAJ --microfilename $MICROTRAJ -o ${OUTPUT}.sh.wts.jpg

## WAITING TIME DISTRIBUTION
WTD_ARGS="--start 1 --final 4 --nsteps 10000000 --max-lagtime 25 --frames-per-unit 1 --unit frames" 
python -m msmhelper waiting-time-dist $WTD_ARGS \
    --filename $MACROTRAJ -o ${OUTPUT}.wtd.jpg
python -m msmhelper waiting-time-dist $WTD_ARGS \
    --filename $MACROTRAJ --microfilename $MICROTRAJ -o ${OUTPUT}.sh.wtd.jpg

## IMPLIED TIMESCALES
IMPL_ARGS="--max-lagtime 25 --frames-per-unit 1 --unit frames" 
python -m msmhelper implied-timescales $IMPL_ARGS \
    --filename $MACROTRAJ -o ${OUTPUT}.impl.jpg
python -m msmhelper implied-timescales $IMPL_ARGS \
    --filename $MACROTRAJ --microfilename $MICROTRAJ -o ${OUTPUT}.sh.impl.jpg

## CKTEST
CKTEST_ARGS="--lagtimes 1 2 3 4 5 --max-time 500 --frames-per-unit 1 --unit frames --grid 2 2" 
python -m msmhelper ck-test $CKTEST_ARGS \
    --filename $MACROTRAJ -o ${OUTPUT}.cktest.jpg
python -m msmhelper ck-test $CKTEST_ARGS \
    --filename $MACROTRAJ --microfilename $MICROTRAJ -o ${OUTPUT}.sh.cktest.jpg
