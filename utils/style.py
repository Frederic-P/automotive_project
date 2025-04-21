import plotly.io as pio

# Create custom template with the desired x-axis and general styles
pio.templates["custom"] = pio.templates["plotly_white"].update({
    "layout": {
        "title": {
            "font": {"size": 24, "color": "black"}
        },
        "xaxis": {
            "title": {"font": {"size": 20, "color": "black"}},
            "tickfont": {"size": 18, "color": "black"},
            "tickangle": -45,  # Rotate the x-axis labels by -45 degrees
            "ticks": 'outside',  # Ticks will be outside the axis
            "ticklen": 10,  # Length of the ticks
            "tickwidth": 2,  # Width of the ticks
            "tickcolor": 'black',  # Color of the ticks
        },
        "yaxis": {
            "title": {"font": {"size": 20, "color": "black"}},
            "tickfont": {"size": 18, "color": "black"}
        },
        "font": {
            "color": "black"
        },
    }
})

# Set the default template globally
pio.templates.default = "custom"
