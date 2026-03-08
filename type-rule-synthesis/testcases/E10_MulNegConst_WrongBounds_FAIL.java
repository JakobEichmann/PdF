public class E10_MulNegConst_WrongBounds_FAIL {
    @Interval(min = 1, max = 3)
    int f;

    public void foo() {
        f = 2;
        // correct is [-6,-2], annotation is wrong order/range
        @Interval(min = -2, max = -6)
        int l = f * (-2); // EXPECT: FAIL
    }
}
