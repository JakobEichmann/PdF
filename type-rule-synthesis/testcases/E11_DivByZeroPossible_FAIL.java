public class E11_DivByZeroPossible_FAIL {
    @Interval(min = 0, max = 2)
    int g;

    public void foo() {
        g = 0;
        @Interval(min = 0, max = 10)
        int l = 10 / g; // EXPECT: FAIL (division by zero possible)
    }
}
