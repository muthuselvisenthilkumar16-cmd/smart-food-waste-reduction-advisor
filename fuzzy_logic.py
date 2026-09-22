import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ============================================================
# CREATE FUZZY SYSTEM
# ============================================================

def create_fuzzy_system():

    # --------------------------------------------------------
    # INPUT VARIABLES
    # --------------------------------------------------------

    leftover = ctrl.Antecedent(
        np.arange(0, 101, 1),
        "leftover"
    )

    demand = ctrl.Antecedent(
        np.arange(0, 101, 1),
        "demand"
    )

    storage = ctrl.Antecedent(
        np.arange(0, 8, 1),
        "storage"
    )

    # --------------------------------------------------------
    # OUTPUT VARIABLE
    # --------------------------------------------------------

    waste_risk = ctrl.Consequent(
        np.arange(0, 101, 1),
        "waste_risk"
    )

    # ========================================================
    # MEMBERSHIP FUNCTIONS
    # ========================================================

    # --------------------------------------------------------
    # LEFTOVER QUANTITY
    # --------------------------------------------------------

    leftover["low"] = fuzz.trimf(
        leftover.universe,
        [0, 0, 40]
    )

    leftover["medium"] = fuzz.trimf(
        leftover.universe,
        [20, 50, 80]
    )

    leftover["high"] = fuzz.trimf(
        leftover.universe,
        [60, 100, 100]
    )

    # --------------------------------------------------------
    # CONSUMPTION DEMAND
    # --------------------------------------------------------

    demand["low"] = fuzz.trimf(
        demand.universe,
        [0, 0, 40]
    )

    demand["medium"] = fuzz.trimf(
        demand.universe,
        [20, 50, 80]
    )

    demand["high"] = fuzz.trimf(
        demand.universe,
        [60, 100, 100]
    )

    # --------------------------------------------------------
    # STORAGE TIME
    # --------------------------------------------------------

    storage["short"] = fuzz.trimf(
        storage.universe,
        [0, 0, 3]
    )

    storage["medium"] = fuzz.trimf(
        storage.universe,
        [1, 3, 5]
    )

    storage["long"] = fuzz.trimf(
        storage.universe,
        [4, 7, 7]
    )

    # --------------------------------------------------------
    # WASTE RISK OUTPUT
    # --------------------------------------------------------

    waste_risk["low"] = fuzz.trimf(
        waste_risk.universe,
        [0, 0, 40]
    )

    waste_risk["medium"] = fuzz.trimf(
        waste_risk.universe,
        [25, 50, 75]
    )

    waste_risk["high"] = fuzz.trimf(
        waste_risk.universe,
        [60, 100, 100]
    )

    # ========================================================
    # FUZZY RULES
    # ========================================================

    # --------------------------------------------------------
    # LOW LEFTOVER
    # --------------------------------------------------------

    rule1 = ctrl.Rule(
        leftover["low"] & demand["low"],
        waste_risk["medium"]
    )

    rule2 = ctrl.Rule(
        leftover["low"] & demand["medium"],
        waste_risk["low"]
    )

    rule3 = ctrl.Rule(
        leftover["low"] & demand["high"],
        waste_risk["low"]
    )

    # --------------------------------------------------------
    # MEDIUM LEFTOVER
    # --------------------------------------------------------

    rule4 = ctrl.Rule(
        leftover["medium"] & demand["low"],
        waste_risk["high"]
    )

    rule5 = ctrl.Rule(
        leftover["medium"] & demand["medium"],
        waste_risk["medium"]
    )

    rule6 = ctrl.Rule(
        leftover["medium"] & demand["high"],
        waste_risk["low"]
    )

    # --------------------------------------------------------
    # HIGH LEFTOVER
    # --------------------------------------------------------

    rule7 = ctrl.Rule(
        leftover["high"] & demand["low"],
        waste_risk["high"]
    )

    rule8 = ctrl.Rule(
        leftover["high"] & demand["medium"],
        waste_risk["medium"]
    )

    rule9 = ctrl.Rule(
        leftover["high"] & demand["high"],
        waste_risk["medium"]
    )

    # --------------------------------------------------------
    # STORAGE EFFECT
    # --------------------------------------------------------

    rule10 = ctrl.Rule(
        storage["long"] & demand["low"],
        waste_risk["high"]
    )

    rule11 = ctrl.Rule(
        storage["long"] & demand["medium"],
        waste_risk["high"]
    )

    rule12 = ctrl.Rule(
        storage["long"] & leftover["high"],
        waste_risk["high"]
    )

    rule13 = ctrl.Rule(
        storage["short"] & leftover["low"],
        waste_risk["low"]
    )

    rule14 = ctrl.Rule(
        storage["short"] & demand["high"],
        waste_risk["low"]
    )

    # --------------------------------------------------------
    # IMPORTANT COVERAGE RULES
    #
    # These make sure situations such as:
    # 10% leftover + 50% demand + 2 days
    # still activate the fuzzy system.
    # --------------------------------------------------------

    rule15 = ctrl.Rule(
        leftover["low"] & demand["medium"] & storage["medium"],
        waste_risk["low"]
    )

    rule16 = ctrl.Rule(
        leftover["low"] & demand["medium"] & storage["short"],
        waste_risk["low"]
    )

    rule17 = ctrl.Rule(
        leftover["medium"] & demand["medium"] & storage["medium"],
        waste_risk["medium"]
    )

    rule18 = ctrl.Rule(
        leftover["high"] & demand["medium"] & storage["medium"],
        waste_risk["medium"]
    )

    # --------------------------------------------------------
    # CREATE CONTROL SYSTEM
    # --------------------------------------------------------

    waste_control = ctrl.ControlSystem([
        rule1,
        rule2,
        rule3,
        rule4,
        rule5,
        rule6,
        rule7,
        rule8,
        rule9,
        rule10,
        rule11,
        rule12,
        rule13,
        rule14,
        rule15,
        rule16,
        rule17,
        rule18
    ])

    return (
        leftover,
        demand,
        storage,
        waste_risk,
        waste_control
    )


# ============================================================
# CALCULATE WASTE RISK
# ============================================================

def calculate_waste_risk(
    leftover_quantity,
    consumption_demand,
    storage_time
):

    (
        leftover,
        demand,
        storage,
        waste_risk,
        waste_control
    ) = create_fuzzy_system()

    simulation = ctrl.ControlSystemSimulation(
        waste_control
    )

    # Set input values
    simulation.input["leftover"] = leftover_quantity
    simulation.input["demand"] = consumption_demand
    simulation.input["storage"] = storage_time

    # Run fuzzy inference
    simulation.compute()

    # --------------------------------------------------------
    # Get defuzzified output
    # --------------------------------------------------------

    if "waste_risk" in simulation.output:

        risk_score = simulation.output["waste_risk"]

    else:
        # Safety fallback for unusual boundary combinations.
        # The fuzzy system remains the primary decision method.
        risk_score = (
            (leftover_quantity * 0.45)
            + ((100 - consumption_demand) * 0.35)
            + ((storage_time / 7) * 100 * 0.20)
        )

        risk_score = max(
            0,
            min(100, risk_score)
        )

    return round(
        float(risk_score),
        2
    )


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(risk_score):

    if risk_score < 40:
        return "Low"

    elif risk_score < 70:
        return "Medium"

    else:
        return "High"


# ============================================================
# GET MEMBERSHIP VALUES
# ============================================================

def get_membership_values(
    leftover_quantity,
    consumption_demand,
    storage_time
):

    (
        leftover,
        demand,
        storage,
        waste_risk,
        waste_control
    ) = create_fuzzy_system()

    # --------------------------------------------------------
    # LEFTOVER MEMBERSHIP
    # --------------------------------------------------------

    leftover_values = {

        "Low": round(
            float(
                fuzz.interp_membership(
                    leftover.universe,
                    leftover["low"].mf,
                    leftover_quantity
                )
            ),
            3
        ),

        "Medium": round(
            float(
                fuzz.interp_membership(
                    leftover.universe,
                    leftover["medium"].mf,
                    leftover_quantity
                )
            ),
            3
        ),

        "High": round(
            float(
                fuzz.interp_membership(
                    leftover.universe,
                    leftover["high"].mf,
                    leftover_quantity
                )
            ),
            3
        )
    }

    # --------------------------------------------------------
    # DEMAND MEMBERSHIP
    # --------------------------------------------------------

    demand_values = {

        "Low": round(
            float(
                fuzz.interp_membership(
                    demand.universe,
                    demand["low"].mf,
                    consumption_demand
                )
            ),
            3
        ),

        "Medium": round(
            float(
                fuzz.interp_membership(
                    demand.universe,
                    demand["medium"].mf,
                    consumption_demand
                )
            ),
            3
        ),

        "High": round(
            float(
                fuzz.interp_membership(
                    demand.universe,
                    demand["high"].mf,
                    consumption_demand
                )
            ),
            3
        )
    }

    # --------------------------------------------------------
    # STORAGE MEMBERSHIP
    # --------------------------------------------------------

    storage_values = {

        "Short": round(
            float(
                fuzz.interp_membership(
                    storage.universe,
                    storage["short"].mf,
                    storage_time
                )
            ),
            3
        ),

        "Medium": round(
            float(
                fuzz.interp_membership(
                    storage.universe,
                    storage["medium"].mf,
                    storage_time
                )
            ),
            3
        ),

        "Long": round(
            float(
                fuzz.interp_membership(
                    storage.universe,
                    storage["long"].mf,
                    storage_time
                )
            ),
            3
        )
    }

    return {
        "leftover": leftover_values,
        "demand": demand_values,
        "storage": storage_values
    }


# ============================================================
# GET FUZZY RULE EVALUATIONS
# ============================================================

def get_rule_evaluations(
    leftover_quantity,
    consumption_demand,
    storage_time
):

    membership = get_membership_values(
        leftover_quantity,
        consumption_demand,
        storage_time
    )

    leftover = membership["leftover"]
    demand = membership["demand"]
    storage = membership["storage"]

    rules = []

    # --------------------------------------------------------
    # R1
    # --------------------------------------------------------

    activation = min(
        leftover["Low"],
        demand["Low"]
    )

    rules.append({
        "rule": "R1",
        "condition": "IF leftover is LOW AND demand is LOW",
        "output": "MEDIUM risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    activation = min(
        leftover["Low"],
        demand["Medium"]
    )

    rules.append({
        "rule": "R2",
        "condition": "IF leftover is LOW AND demand is MEDIUM",
        "output": "LOW risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R3
    # --------------------------------------------------------

    activation = min(
        leftover["Low"],
        demand["High"]
    )

    rules.append({
        "rule": "R3",
        "condition": "IF leftover is LOW AND demand is HIGH",
        "output": "LOW risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R4
    # --------------------------------------------------------

    activation = min(
        leftover["Medium"],
        demand["Low"]
    )

    rules.append({
        "rule": "R4",
        "condition": "IF leftover is MEDIUM AND demand is LOW",
        "output": "HIGH risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R5
    # --------------------------------------------------------

    activation = min(
        leftover["Medium"],
        demand["Medium"]
    )

    rules.append({
        "rule": "R5",
        "condition": "IF leftover is MEDIUM AND demand is MEDIUM",
        "output": "MEDIUM risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R6
    # --------------------------------------------------------

    activation = min(
        leftover["Medium"],
        demand["High"]
    )

    rules.append({
        "rule": "R6",
        "condition": "IF leftover is MEDIUM AND demand is HIGH",
        "output": "LOW risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R7
    # --------------------------------------------------------

    activation = min(
        leftover["High"],
        demand["Low"]
    )

    rules.append({
        "rule": "R7",
        "condition": "IF leftover is HIGH AND demand is LOW",
        "output": "HIGH risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R8
    # --------------------------------------------------------

    activation = min(
        leftover["High"],
        demand["Medium"]
    )

    rules.append({
        "rule": "R8",
        "condition": "IF leftover is HIGH AND demand is MEDIUM",
        "output": "MEDIUM risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R9
    # --------------------------------------------------------

    activation = min(
        leftover["High"],
        demand["High"]
    )

    rules.append({
        "rule": "R9",
        "condition": "IF leftover is HIGH AND demand is HIGH",
        "output": "MEDIUM risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R10
    # --------------------------------------------------------

    activation = min(
        storage["Long"],
        demand["Low"]
    )

    rules.append({
        "rule": "R10",
        "condition": "IF storage is LONG AND demand is LOW",
        "output": "HIGH risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R11
    # --------------------------------------------------------

    activation = min(
        storage["Long"],
        demand["Medium"]
    )

    rules.append({
        "rule": "R11",
        "condition": "IF storage is LONG AND demand is MEDIUM",
        "output": "HIGH risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R12
    # --------------------------------------------------------

    activation = min(
        storage["Long"],
        leftover["High"]
    )

    rules.append({
        "rule": "R12",
        "condition": "IF storage is LONG AND leftover is HIGH",
        "output": "HIGH risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R13
    # --------------------------------------------------------

    activation = min(
        storage["Short"],
        leftover["Low"]
    )

    rules.append({
        "rule": "R13",
        "condition": "IF storage is SHORT AND leftover is LOW",
        "output": "LOW risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R14
    # --------------------------------------------------------

    activation = min(
        storage["Short"],
        demand["High"]
    )

    rules.append({
        "rule": "R14",
        "condition": "IF storage is SHORT AND demand is HIGH",
        "output": "LOW risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R15
    # --------------------------------------------------------

    activation = min(
        leftover["Low"],
        demand["Medium"],
        storage["Medium"]
    )

    rules.append({
        "rule": "R15",
        "condition": (
            "IF leftover is LOW AND "
            "demand is MEDIUM AND "
            "storage is MEDIUM"
        ),
        "output": "LOW risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R16
    # --------------------------------------------------------

    activation = min(
        leftover["Low"],
        demand["Medium"],
        storage["Short"]
    )

    rules.append({
        "rule": "R16",
        "condition": (
            "IF leftover is LOW AND "
            "demand is MEDIUM AND "
            "storage is SHORT"
        ),
        "output": "LOW risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R17
    # --------------------------------------------------------

    activation = min(
        leftover["Medium"],
        demand["Medium"],
        storage["Medium"]
    )

    rules.append({
        "rule": "R17",
        "condition": (
            "IF leftover is MEDIUM AND "
            "demand is MEDIUM AND "
            "storage is MEDIUM"
        ),
        "output": "MEDIUM risk",
        "activation": round(activation, 3)
    })

    # --------------------------------------------------------
    # R18
    # --------------------------------------------------------

    activation = min(
        leftover["High"],
        demand["Medium"],
        storage["Medium"]
    )

    rules.append({
        "rule": "R18",
        "condition": (
            "IF leftover is HIGH AND "
            "demand is MEDIUM AND "
            "storage is MEDIUM"
        ),
        "output": "MEDIUM risk",
        "activation": round(activation, 3)
    })

    return rules


# ============================================================
# COMPLETE FUZZY ANALYSIS
# ============================================================

def analyze_fuzzy_system(
    leftover_quantity,
    consumption_demand,
    storage_time
):

    # --------------------------------------------------------
    # Calculate fuzzy risk
    # --------------------------------------------------------

    risk_score = calculate_waste_risk(
        leftover_quantity,
        consumption_demand,
        storage_time
    )

    # --------------------------------------------------------
    # Determine risk level
    # --------------------------------------------------------

    risk_level = get_risk_level(
        risk_score
    )

    # --------------------------------------------------------
    # Fuzzification
    # --------------------------------------------------------

    membership_values = get_membership_values(
        leftover_quantity,
        consumption_demand,
        storage_time
    )

    # --------------------------------------------------------
    # Rule evaluation
    # --------------------------------------------------------

    rule_evaluations = get_rule_evaluations(
        leftover_quantity,
        consumption_demand,
        storage_time
    )

    # --------------------------------------------------------
    # Return complete result
    # --------------------------------------------------------

    return {

        "risk_score": risk_score,

        "risk_level": risk_level,

        "membership_values": membership_values,

        "rule_evaluations": rule_evaluations,

        "defuzzified_score": risk_score
    }