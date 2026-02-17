from vggt_jax.models.vggt.modeling import create_vggt_from_torch_checkpoint


def main() -> None:
    _, report = create_vggt_from_torch_checkpoint(
        checkpoint_path="weights/model.pt",
        strict=True,
        include_track=True,
    )

    print("total_source_keys", report.total_source_keys)
    print("mapped_keys", report.mapped_keys)
    print("loaded_keys", report.loaded_keys)
    print("skipped_unmapped", report.skipped_unmapped)
    print("skipped_missing_target", report.skipped_missing_target)
    print("skipped_shape_mismatch", report.skipped_shape_mismatch)
    print("missing_target_leaves", report.missing_target_leaves)

    if report.loaded_keys != report.total_source_keys:
        raise RuntimeError(
            f"Expected all source keys loaded, got loaded={report.loaded_keys}, total={report.total_source_keys}"
        )
    if report.skipped_unmapped or report.skipped_missing_target or report.skipped_shape_mismatch:
        raise RuntimeError(
            "Strict loading incomplete: "
            f"unmapped={report.skipped_unmapped}, "
            f"missing_target={report.skipped_missing_target}, "
            f"shape_mismatch={report.skipped_shape_mismatch}"
        )

    print("status PASS")


if __name__ == "__main__":
    main()
