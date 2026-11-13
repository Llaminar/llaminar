/**
 * @file CalibrationCoordinateIdentityContract.inc
 * @brief Reject same-sized foreign calibration ledgers before physical work.
 *
 * The actual controller borrows the existing prepared-transport fixture. Equal
 * row counts cannot substitute for exact endpoint/layer identities, while input
 * ordering is normalized once by the original ledger constructor.
 */
namespace llaminar2::test
{
TEST(MoEOverlayEconomyCalibrationController, RejectsSameSizedForeignCoordinatesBeforeTransfer)
{
    ProfileFixture fixture;
    const auto original = fixture.planner->requiredCoordinates();
    ASSERT_EQ(original.size(), 2u);
    for (const bool change_layer : {false, true})
    {
        auto changed = original;
        for (auto &coordinate : changed)
        {
            if (change_layer)
                ++coordinate.layer;
            else
            {
                ++coordinate.source_participant;
                ++coordinate.destination_participant;
            }
        }
        auto foreign = std::make_shared<MoEOverlayMigrationMeasurementLedger>(
            MoEOverlayMigrationMeasurementLedger::Config{
                .required_coordinates = changed,
                .warmup_samples_per_coordinate = 0,
                .measured_samples_per_coordinate = 3,
                .measurement_identity = "recorded-foreign-coordinate-negative",
            });
        ASSERT_EQ(foreign->coordinateCount(), fixture.ledger->coordinateCount());
        EXPECT_THROW((void)MoEOverlayEconomyCalibrationController(
            MoEOverlayEconomyCalibrationController::Config{
                .planner = fixture.planner,
                .ledger = foreign,
                .journal = fixture.journal,
                .transport = fixture.transport,
            }), std::invalid_argument);
        EXPECT_EQ(foreign->stats().completed_waves_observed, 0u);
        EXPECT_EQ(fixture.factory.observations->waves_prepared, 0u);
        EXPECT_EQ(fixture.factory.observations->projection_polls, 0u);
    }

    auto reordered = original;
    std::reverse(reordered.begin(), reordered.end());
    auto matching = std::make_shared<MoEOverlayMigrationMeasurementLedger>(
        MoEOverlayMigrationMeasurementLedger::Config{
            .required_coordinates = reordered,
            .warmup_samples_per_coordinate = 0,
            .measured_samples_per_coordinate = 3,
            .measurement_identity = "recorded-canonical-order-positive",
        });
    EXPECT_EQ(matching->requiredCoordinates(), original);
    EXPECT_NO_THROW((void)MoEOverlayEconomyCalibrationController(
        MoEOverlayEconomyCalibrationController::Config{
            .planner = fixture.planner,
            .ledger = matching,
            .journal = fixture.journal,
            .transport = fixture.transport,
        }));
    EXPECT_EQ(fixture.factory.observations->waves_prepared, 0u);
}
}
