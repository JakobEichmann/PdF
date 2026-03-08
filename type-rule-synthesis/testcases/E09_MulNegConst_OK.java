public class E09_MulNegConst_OK {
    @Interval(min = 1, max = 3)
    int f;

    public void foo() {
        f = 2;
        // f * (-2) in [-6,-2]
        @Interval(min = -6, max = -2)
        int l = f * (-2); // EXPECT: OK (needs sign-aware scaling)
    }
}
