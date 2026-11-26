import { expect, test } from "@playwright/test";

test.describe("AutoSpec AI End-to-End", () => {
  test("landing page navigates to live demo workspace and executes query", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Precision Engineering.")).toBeVisible();

    await page.getByRole("link", { name: /Try Live Demo/i }).click();
    await expect(page).toHaveURL(/\/workspace/);

    const input = page.getByLabel("Specification query");
    await input.fill("What is the torque for front brake caliper guide pin bolts on 2022 Apex M400?");
    await page.getByRole("button", { name: "Search" }).click();

    await expect(page.getByText("37 Nm")).toBeVisible();
    await expect(page.getByText("Verified")).toBeVisible();
  });

  test("vehicle comparison and evaluation analytics pages load cleanly", async ({ page }) => {
    await page.goto("/compare");
    await expect(page.getByText("Vehicle Specification Comparison")).toBeVisible();

    await page.goto("/analytics");
    await expect(page.getByText("Evaluation & Retrieval Analytics")).toBeVisible();
  });
});

