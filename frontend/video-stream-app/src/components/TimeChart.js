import React from 'react';
import { Line } from '@ant-design/charts';
import { useEffect, useState } from 'react';
import axios from 'axios';

const TimeChart = ({ height }) => {
    const [data, setData] = useState([]);

    useEffect(() => {
      axios.get('http://localhost:5000/get_records_data')
        .then(response => {
          const transformedData = response.data.map(item => ({
            dateTime: new Date(`${item.startTime}`),
            value: item.length,
          }));
          setData(transformedData);
        })
        .catch(error => console.error('Error:', error));
    }, []);

    const config = {
        data,
        height, 
        xField: 'dateTime',
        yField: 'value',
        seriesField: 'name',
        xAxis: {
            type: 'time',
            title: {
                text: 'Date and Time',
            },
        },
        yAxis: {
            title: {
                text: 'Duration',
            },
        },
        title: {
            text: 'Duration for Sessions',  // Add your title here
        }
    }
    return (
        <Line title="Duration for Sessions" {...config} />
    );
}

export default TimeChart;
