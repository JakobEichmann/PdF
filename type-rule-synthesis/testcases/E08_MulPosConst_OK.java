public class E08_MulPosConst_OK {
    @Interval(min = 1, max = 3)
    int f;

    public void foo() {
        f = 2;
        // f * 2 in [2,6]
        @Interval(min = 2, max = 6)
        int l = f * 2; // EXPECT: OK (needs scaling rule)
    }
}
