use serde::Deserialize;
use time::OffsetDateTime;

pub fn deserialize_iso8601_timestamp_to_unix_timestamp<'de, D>(
    deserializer: D,
) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let iso8601_str = String::deserialize(deserializer)?;

    // Parse ISO8601 string to OffsetDateTime
    let datetime = OffsetDateTime::parse(
        &iso8601_str,
        &time::format_description::well_known::Iso8601::DEFAULT,
    )
    .map_err(serde::de::Error::custom)?;

    // Convert to Unix timestamp
    Ok(datetime.unix_timestamp() as u64)
}
