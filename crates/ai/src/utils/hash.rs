pub fn short_hash(input: &str) -> String {
    let mut h1 = 0xdead_beefu32;
    let mut h2 = 0x41c6_ce57u32;
    for ch in input.encode_utf16() {
        let ch = u32::from(ch);
        h1 = (h1 ^ ch).wrapping_mul(2_654_435_761);
        h2 = (h2 ^ ch).wrapping_mul(1_597_334_677);
    }
    h1 = ((h1 ^ (h1 >> 16)).wrapping_mul(2_246_822_507))
        ^ ((h2 ^ (h2 >> 13)).wrapping_mul(3_266_489_909));
    h2 = ((h2 ^ (h2 >> 16)).wrapping_mul(2_246_822_507))
        ^ ((h1 ^ (h1 >> 13)).wrapping_mul(3_266_489_909));
    format!("{}{}", to_base36(h2), to_base36(h1))
}

fn to_base36(mut value: u32) -> String {
    if value == 0 {
        return "0".to_string();
    }
    let mut digits = Vec::new();
    while value > 0 {
        let digit = (value % 36) as u8;
        digits.push(match digit {
            0..=9 => b'0' + digit,
            _ => b'a' + (digit - 10),
        });
        value /= 36;
    }
    digits.reverse();
    String::from_utf8(digits).expect("base36 digits are valid utf8")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_upstream_short_hash() {
        assert_eq!(short_hash(""), "k4n83c7h0j2b");
        assert_eq!(short_hash("hello"), "1h6qa0qrowduu");
        assert_eq!(short_hash("foreign-tool-call-id"), "1r12n89219uo");
        assert_eq!(short_hash("😀"), "13wj7r7usi372");
    }
}
