/**
 annotation Remainder(int remainder, int modulus) int
    :<==> "java.lang.Math.floorMod(§subject§, §modulus§) == §remainder§"
    for "0 <= §remainder§ && §remainder§ < §modulus§";
*/

public class RemainderProperties {

    public static void foo1(
            @Remainder(remainder="1", modulus="6") int arg0,
            @Remainder(remainder="4", modulus="6") int arg1) {
        // :: error: assignment.type.incompatible
        @Remainder(remainder="1", modulus="3") int a = arg0;
        // :: error: assignment.type.incompatible
        a = arg1;
    }
}

/**
Expected result:

v0 : @Remainder(n0, m0)
n1 = n0 + m0
---
v0 : @Remainder(n1, m0)

v0 : @Remainder(n0, m0)
n1 = n0 - m0
---
v0 : @Remainder(n1, m0)

v0 : @Remainder(n0, m0)
m0 = m1 * k
---
v0 : @Remainder(n0, m1)
*/
